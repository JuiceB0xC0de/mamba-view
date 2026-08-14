"""Precision timing recorder, shared by the shot runner and any camera.

Lifted from Camera 7's ``PrecisionTimingRecorder`` so the runner can profile
every hook without importing a camera.  Same schema, so its output is directly
comparable to the Camera 7 timing captures.

CUDA events are resolved once per completed step rather than at every boundary,
so profiling does not insert a synchronize into the middle of a measured
region.  Host time and GPU time are reported separately because the interesting
failure mode here is CPU-bound work (compression, full-vocabulary metrics)
hiding behind what looks like GPU cost.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path


def _sha256_path(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PrecisionTimingRecorder:
    SCHEMA_VERSION = 1

    def __init__(self, torch_module=None, *, enabled=True):
        self.torch = torch_module
        self.enabled = bool(enabled)
        self.events = []
        self._pending_cuda = []
        self._next_sequence = 0

    def host_mark(self) -> int:
        return time.perf_counter_ns()

    def _new_event(
        self, name, started_ns, finished_ns, dimensions, *, cuda_requested=False
    ):
        event = {
            "sequence": self._next_sequence,
            "name": str(name),
            "host_started_ns": int(started_ns),
            "host_finished_ns": int(finished_ns),
            "host_elapsed_ns": int(finished_ns - started_ns),
            "cuda_requested": bool(cuda_requested),
            "cuda_elapsed_ns": None,
            **dimensions,
        }
        self._next_sequence += 1
        self.events.append(event)
        return event

    def record_host_since(self, name: str, started_ns: int, **dimensions):
        if not self.enabled:
            return None
        return self._new_event(name, started_ns, time.perf_counter_ns(), dimensions)

    def _cuda_ready(self) -> bool:
        return bool(
            self.enabled and self.torch is not None and self.torch.cuda.is_available()
        )

    @contextmanager
    def span(self, name: str, *, cuda: bool = False, **dimensions):
        if not self.enabled:
            yield
            return
        start_cuda = end_cuda = None
        if cuda and self._cuda_ready():
            start_cuda = self.torch.cuda.Event(enable_timing=True)
            end_cuda = self.torch.cuda.Event(enable_timing=True)
            start_cuda.record()
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            finished_ns = time.perf_counter_ns()
            if end_cuda is not None:
                end_cuda.record()
            event = self._new_event(
                name, started_ns, finished_ns, dimensions, cuda_requested=cuda
            )
            if start_cuda is not None:
                self._pending_cuda.append((event, start_cuda, end_cuda))

    def resolve_cuda(self) -> None:
        if not self._pending_cuda:
            return
        self.torch.cuda.synchronize()
        for event, start_cuda, end_cuda in self._pending_cuda:
            event["cuda_elapsed_ns"] = int(
                round(float(start_cuda.elapsed_time(end_cuda)) * 1_000_000.0)
            )
        self._pending_cuda.clear()

    @staticmethod
    def _percentile(values, q):
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        frac = position - lo
        return int(round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac))

    def summary(self) -> dict:
        grouped = {}
        for event in self.events:
            grouped.setdefault(event["name"], []).append(event)
        rows = []
        for name, events in sorted(grouped.items()):
            host = [e["host_elapsed_ns"] for e in events]
            cuda = [
                e["cuda_elapsed_ns"] for e in events if e["cuda_elapsed_ns"] is not None
            ]
            rows.append(
                {
                    "name": name,
                    "count": len(events),
                    "host_total_ns": sum(host),
                    "host_mean_ns": int(round(sum(host) / len(host))),
                    "host_p50_ns": self._percentile(host, 0.50),
                    "host_p95_ns": self._percentile(host, 0.95),
                    "host_max_ns": max(host),
                    "cuda_count": len(cuda),
                    "cuda_total_ns": sum(cuda) if cuda else None,
                    "cuda_mean_ns": (
                        int(round(sum(cuda) / len(cuda))) if cuda else None
                    ),
                    "cuda_p50_ns": self._percentile(cuda, 0.50),
                    "cuda_p95_ns": self._percentile(cuda, 0.95),
                    "cuda_max_ns": max(cuda) if cuda else None,
                }
            )
        return {
            "timing_schema_version": self.SCHEMA_VERSION,
            "clock": "time.perf_counter_ns",
            "cuda_clock": "torch.cuda.Event elapsed_time",
            "cuda_resolution_policy": "one torch.cuda.synchronize per step",
            "event_count": len(self.events),
            "groups": rows,
        }

    def ranked(self, top=20) -> list:
        """Groups by total host time, largest first.  For printing."""
        rows = sorted(
            self.summary()["groups"],
            key=lambda row: row["host_total_ns"],
            reverse=True,
        )
        return rows[:top]

    def write(self, out_dir, context: dict) -> dict:
        self.resolve_cuda()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        events_path = out_dir / "timing_events.jsonl"
        summary_path = out_dir / "timing_summary.json"
        ordered = sorted(self.events, key=lambda e: e["sequence"])
        events_path.write_text(
            "".join(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                for event in ordered
            )
        )
        payload = {**self.summary(), "context": context}
        summary_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        hashes = {
            events_path.name: _sha256_path(events_path),
            summary_path.name: _sha256_path(summary_path),
        }
        sums_path = out_dir / "SHA256SUMS"
        sums_path.write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
        )
        return {
            "files": [events_path.name, summary_path.name, sums_path.name],
            "hashes": hashes,
            "event_count": len(ordered),
        }
