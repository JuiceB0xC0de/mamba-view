"""Training-time time/retention observer for Mamba-3 (camera 2, steps 20-25).

Modal-free by design so every mechanism is CPU-testable. The Modal trainer
(modal_time_observer.py) imports this module; nothing here imports modal,
wandb, or CUDA-only code.

Capture family: time/retention ONLY. The observer hooks each Mamba-3 block's
``mixer.in_proj`` FORWARD PROJECTION OUTPUT (not gradients) and clones exactly
four small slices -- dd_dt, dd_A, trap, angles -- as defined by the canonical
``InProjSpec`` split [z, x, B, C, dd_dt, dd_A, trap, angles].

All recurrence math comes from ``analysis.mamba3_core.recurrence_quantities``
and all phase math from ``analysis.reference_recurrence.phase_details``. This
module never transcribes those formulas. The invalid raw-angle-slice RoPE
approach (tombstoned by INJECTION_METRICS_VALID=False in
analysis/capture_stage_b.py) is not revived: raw angles are stored as raw
basis only and every phase quantity is produced by phase_details.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# One canonical module object. reference_recurrence internally imports
# "mamba3_core" as a top-level name; pre-registering the analysis module under
# that name prevents a second module object with divergent identity, which
# would silently break monkeypatching and reuse assertions.
from analysis import mamba3_core as _mamba3_core  # noqa: E402

_existing = sys.modules.get("mamba3_core")
if _existing is not None and _existing is not _mamba3_core:
    raise ImportError(
        "mamba3_core is already loaded under a conflicting module identity; "
        "refusing to continue with two copies of the canonical module"
    )
sys.modules["mamba3_core"] = _mamba3_core

from analysis import reference_recurrence as _reference_recurrence  # noqa: E402

OBSERVER_SCHEMA_VERSION = 1
OBSERVER_FAMILY = "time_retention"
CAPTURE_START_STEP = 20
CAPTURE_END_STEP = 25
CAPTURE_WINDOW = frozenset(range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1))
PHASES = ("pre_update", "post_update")

# in_proj slices captured by the hook; everything else in the projection is out
# of scope for the time/retention family.
HOOKED_SLICES = ("dd_dt", "dd_A", "trap", "angles")

# Exact storage names. Raw sufficient basis first, then canonical derived.
RAW_NAMES = ("dd_dt", "dd_A", "trap_raw", "angle_raw", "dt_bias", "A_floor")
DERIVED_NAMES = (
    "Delta", "A", "ADT", "alpha", "lambda", "gamma", "shifted_gamma",
    "trap_scale", "local_halflife", "subjective_time",
    "phase_increment", "phase_unwrapped", "phase_wrapped", "phase_carry_out",
)

_MIXER_ATTRS = ("in_proj", "dt_bias", "B_bias", "C_bias", "B_norm", "C_norm", "out_proj")


class ObserverError(RuntimeError):
    """Base class for observer failures. Always fail loudly, never impute."""


class NonFiniteCaptureError(ObserverError):
    """A captured or derived value is non-finite. Per the MaskedMoments
    doctrine, non-finite observations are never silently replaced or averaged
    through; the run dies here."""


class GeometryMismatchError(ObserverError):
    """Observed layers do not share compatible in_proj split geometry."""


# ---------------------------------------------------------------------------
# derivation helpers (definitions only; no recurrence/phase math transcribed)
# ---------------------------------------------------------------------------


def subjective_time(delta: torch.Tensor) -> torch.Tensor:
    """tau = inclusive cumsum of Delta over token position (dim 0)."""
    return torch.cumsum(delta, dim=0)


def decay_cost(adt: torch.Tensor) -> torch.Tensor:
    """Positive decay cost, defined explicitly as -ADT. Summary-only name."""
    return -adt


def token_ids_sha256(ids) -> str:
    """SHA-256 of the token IDs serialized as explicit little-endian int64."""
    return hashlib.sha256(np.asarray(list(ids), dtype="<i8").tobytes()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# RNG discipline
# ---------------------------------------------------------------------------


def rng_snapshot() -> dict:
    snap = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state_all()
    return snap


def rng_restore(snap: dict) -> None:
    random.setstate(snap["python"])
    torch.set_rng_state(snap["torch_cpu"])
    if "torch_cuda" in snap and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap["torch_cuda"])


# ---------------------------------------------------------------------------
# mixer discovery
# ---------------------------------------------------------------------------


def discover_mixers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Find every Mamba-3 mixer module, in deterministic registration order."""
    found = [
        (name, mod)
        for name, mod in model.named_modules()
        if all(hasattr(mod, a) for a in _MIXER_ATTRS)
    ]
    if not found:
        raise ObserverError(
            f"no Mamba-3 mixers found: no module exposes all of {_MIXER_ATTRS}"
        )
    return found


# ---------------------------------------------------------------------------
# capture containers
# ---------------------------------------------------------------------------


@dataclass
class StepCapture:
    """Raw hook output for one (step, phase) forward. Torch tensors, detached."""

    step: int
    phase: str
    raw: dict = field(default_factory=dict)            # layer idx -> {slice: tensor}
    source_dtypes: dict = field(default_factory=dict)  # layer idx -> str
    dt_bias: dict = field(default_factory=dict)        # layer idx -> float32 tensor
    dt_bias_source_dtype: dict = field(default_factory=dict)
    a_floor: dict = field(default_factory=dict)        # layer idx -> float


@dataclass
class DerivedCapture:
    """Finalized numpy arrays plus per-tensor records for one (step, phase)."""

    step: int
    phase: str
    arrays: dict                 # npz key -> np.ndarray
    tensor_records: list         # manifest records, deterministic order
    token_ids: list
    token_sha256: str
    sequence_length: int
    raw_bytes: int

    @property
    def filename(self) -> str:
        return f"step_{self.step:06d}_{self.phase}.npz"


# ---------------------------------------------------------------------------
# observer
# ---------------------------------------------------------------------------


class TimeRetentionObserver:
    """Temporary-forward-hook observer for the Mamba-3 time/retention family.

    Hooks exist ONLY inside an active ``capture`` context; registration and
    removal are bracketed by try/finally so exceptions cannot leak a hook.
    """

    def __init__(self, model: torch.nn.Module, capture_steps=None):
        # Resolved here rather than as a default argument: defaults bind at
        # import, which made the capture window impossible to retarget.
        capture_steps = CAPTURE_WINDOW if capture_steps is None else capture_steps
        self.capture_steps = frozenset(int(s) for s in capture_steps)
        self.layers = discover_mixers(model)
        self.specs = []
        for name, mixer in self.layers:
            spec = _mamba3_core.InProjSpec.from_mixer(mixer)
            spec.assert_against(int(mixer.in_proj.weight.shape[0]))
            self.specs.append(spec)

        geoms = [
            (s.d_inner, s.d_state, s.ngroups, s.mimo_rank, s.nheads, s.headdim,
             s.n_rope_angles)
            for s in self.specs
        ]
        if len(set(geoms)) != 1:
            detail = {self.layers[i][0]: geoms[i] for i in range(len(geoms))}
            raise GeometryMismatchError(
                f"in_proj split geometry differs across observed layers: {detail}"
            )
        s0 = self.specs[0]
        # Discovered at runtime from realized shapes; never hardcoded.
        self.geometry = {
            "num_layers": len(self.layers),
            "d_inner": s0.d_inner,
            "d_state": s0.d_state,
            "ngroups": s0.ngroups,
            "mimo_rank": s0.mimo_rank,
            "nheads": s0.nheads,          # Mamba recurrence heads
            "headdim": s0.headdim,
            "n_rope_angles": s0.n_rope_angles,
            "layer_module_paths": [name for name, _ in self.layers],
        }
        for name, mixer in self.layers:
            if not hasattr(mixer, "A_floor"):
                raise ObserverError(f"mixer {name} has no A_floor attribute")

    def should_capture(self, step: int) -> bool:
        return int(step) in self.capture_steps

    def hooks_active(self) -> bool:
        return any(
            len(mixer.in_proj._forward_hooks) > 0 for _, mixer in self.layers
        )

    def _make_hook(self, layer_idx: int, spec, cap: StepCapture):
        def hook(module, args, output):
            if layer_idx in cap.raw:
                raise ObserverError(
                    f"layer {layer_idx} observed twice within one capture "
                    f"(step {cap.step}, {cap.phase}); unexpected re-entry"
                )
            out = output.detach()
            slices = {}
            for k in HOOKED_SLICES:
                lo, hi = spec.bounds[k]
                # clone() releases the full projection storage; keeping a view
                # would retain the entire in_proj output tensor.
                slices[k] = out[..., lo:hi].clone()
            cap.raw[layer_idx] = slices
            cap.source_dtypes[layer_idx] = str(output.dtype)
            return None

        return hook

    @contextmanager
    def capture(self, step: int, phase: str):
        if not self.should_capture(step):
            raise ObserverError(
                f"capture requested for step {step}, outside window "
                f"{sorted(self.capture_steps)}"
            )
        if phase not in PHASES:
            raise ObserverError(f"unknown phase {phase!r}; expected one of {PHASES}")
        cap = StepCapture(step=int(step), phase=phase)
        for idx, (_, mixer) in enumerate(self.layers):
            cap.dt_bias[idx] = mixer.dt_bias.detach().float().clone()
            cap.dt_bias_source_dtype[idx] = str(mixer.dt_bias.dtype)
            cap.a_floor[idx] = float(mixer.A_floor)
        handles = []
        try:
            for idx, (_, mixer) in enumerate(self.layers):
                handles.append(
                    mixer.in_proj.register_forward_hook(
                        self._make_hook(idx, self.specs[idx], cap)
                    )
                )
            yield cap
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------
    # derivation
    # ------------------------------------------------------------------

    def derive(self, cap: StepCapture, token_ids) -> DerivedCapture:
        """Compute canonical derived quantities and package numpy arrays.

        Raises NonFiniteCaptureError on ANY non-finite raw or derived value.
        """
        expected = set(range(len(self.layers)))
        if set(cap.raw.keys()) != expected:
            raise ObserverError(
                f"step {cap.step} {cap.phase}: captured layers "
                f"{sorted(cap.raw)} != expected {sorted(expected)}"
            )

        arrays: dict = {}
        records: list = []
        raw_bytes = 0
        seq_len = None

        def _admit(layer_idx, name, tensor, source_dtype):
            nonlocal raw_bytes
            t = tensor.detach().to(torch.float32).cpu()
            finite = int(torch.isfinite(t).sum())
            total = t.numel()
            if finite != total:
                raise NonFiniteCaptureError(
                    f"step {cap.step} {cap.phase} layer {layer_idx} tensor "
                    f"{name}: {total - finite} non-finite of {total}"
                )
            arr = t.numpy()
            key = f"layer_{layer_idx:02d}/{name}"
            arrays[key] = arr
            raw_bytes += arr.nbytes
            records.append({
                "key": key,
                "layer": layer_idx,
                "name": name,
                "shape": list(arr.shape),
                "source_dtype": source_dtype,
                "storage_dtype": "float32",
                "finite_count": finite,
                "nonfinite_count": total - finite,
                "min": float(t.min()) if total else None,
                "max": float(t.max()) if total else None,
                "mean": float(t.mean()) if total else None,
                "std": float(t.std()) if total > 1 else 0.0,
                "raw_bytes": int(arr.nbytes),
            })

        for idx in range(len(self.layers)):
            slices = cap.raw[idx]
            src_dtype = cap.source_dtypes[idx]

            def _flat(name):
                t = slices[name]
                if t.ndim != 3 or t.shape[0] != 1:
                    raise ObserverError(
                        f"step {cap.step} {cap.phase} layer {idx} slice {name}: "
                        f"expected shape (1, T, width), got {tuple(t.shape)}"
                    )
                return t[0].float()

            dd_dt = _flat("dd_dt")          # (T, nheads)
            dd_A = _flat("dd_A")            # (T, nheads)
            trap_raw = _flat("trap")        # (T, nheads) RAW logit
            angle_raw = _flat("angles")     # (T, n_rope_angles), NOT effective phase
            if seq_len is None:
                seq_len = int(dd_dt.shape[0])

            dt_bias = cap.dt_bias[idx]
            a_floor = cap.a_floor[idx]

            # Canonical shared implementations. Never reimplemented here.
            rec = _mamba3_core.recurrence_quantities(
                dd_dt, dd_A, trap_raw, dt_bias, A_floor=a_floor
            )
            tau = subjective_time(rec["Delta"])
            # Full independent sequences per GSM8K example => carried=None.
            ph = _reference_recurrence.phase_details(
                angle_raw, rec["Delta"], carried=None
            )

            _admit(idx, "dd_dt", dd_dt, src_dtype)
            _admit(idx, "dd_A", dd_A, src_dtype)
            _admit(idx, "trap_raw", trap_raw, src_dtype)
            _admit(idx, "angle_raw", angle_raw, src_dtype)
            _admit(idx, "dt_bias", dt_bias, cap.dt_bias_source_dtype[idx])
            # 0-d scalar: float(arr) round-trips and manifest records shape []
            _admit(idx, "A_floor", torch.tensor(a_floor), "python_float")

            _admit(idx, "Delta", rec["Delta"], "float32")
            _admit(idx, "A", rec["A"], "float32")
            _admit(idx, "ADT", rec["ADT"], "float32")
            _admit(idx, "alpha", rec["alpha"], "float32")
            _admit(idx, "lambda", rec["lambda"], "float32")
            _admit(idx, "gamma", rec["gamma"], "float32")
            _admit(idx, "shifted_gamma", rec["shifted_gamma"], "float32")
            _admit(idx, "trap_scale", rec["trap_scale"], "float32")
            _admit(idx, "local_halflife", rec["local_halflife"], "float32")
            _admit(idx, "subjective_time", tau, "float32")
            _admit(idx, "phase_increment", ph["increment"], "float32")
            _admit(idx, "phase_unwrapped", ph["unwrapped"], "float32")
            _admit(idx, "phase_wrapped", ph["wrapped"], "float32")
            _admit(idx, "phase_carry_out", ph["carry_out"], "float32")

        ids = [int(i) for i in token_ids]
        ids_arr = np.asarray(ids, dtype="<i8")
        arrays["token_ids"] = ids_arr
        raw_bytes += ids_arr.nbytes
        records.append({
            "key": "token_ids",
            "layer": None,
            "name": "token_ids",
            "shape": list(ids_arr.shape),
            "source_dtype": "int64",
            "storage_dtype": "int64",
            "finite_count": int(ids_arr.size),
            "nonfinite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "raw_bytes": int(ids_arr.nbytes),
        })

        return DerivedCapture(
            step=cap.step,
            phase=cap.phase,
            arrays=arrays,
            tensor_records=records,
            token_ids=ids,
            token_sha256=token_ids_sha256(ids),
            sequence_length=seq_len if seq_len is not None else 0,
            raw_bytes=raw_bytes,
        )


# ---------------------------------------------------------------------------
# paired post-update replay
# ---------------------------------------------------------------------------


def post_update_replay(model, observer: TimeRetentionObserver, step: int,
                       inputs, targets, loss_fn):
    """Replay identical inputs under theta_step with capture, no side effects.

    RNG is snapshotted immediately before the replay and restored in finally.
    Gradients, parameters, optimizer state, and train/eval mode are untouched.
    Returns (post_loss_value, DerivedCapture-ready StepCapture).
    """
    snap = rng_snapshot()
    try:
        with torch.no_grad():
            with observer.capture(step, "post_update") as cap:
                out = model(inputs)
            loss = loss_fn(out, targets)
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise NonFiniteCaptureError(
                f"step {step} post_update: non-finite replay loss {loss_value}"
            )
        return loss_value, cap
    finally:
        rng_restore(snap)


# ---------------------------------------------------------------------------
# artifact writing -- acyclic finalization order:
#   (a) NPZ files  (b) storage_report.json  (c) hash NPZs + storage report
#   (d) manifest.json embedding those hashes  (e) SHA256SUMS covering
#       NPZs + storage_report.json + manifest.json
# ---------------------------------------------------------------------------


def _write_npz_deterministic(path: Path, arrays: dict) -> None:
    """Write a compressed NPZ with fixed ZIP metadata so identical arrays
    always yield byte-identical archives (no wall-clock timestamps, fixed
    member order, fixed permissions, fixed deflate level). No pickle, no
    object arrays."""
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_STORED
    ) as zf:
        for key in sorted(arrays):
            arr = np.asarray(arrays[key])
            if arr.dtype == object:
                raise ObserverError(f"object array refused for key {key}")
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr, allow_pickle=False)
            zinfo = zipfile.ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zinfo.compress_type = zipfile.ZIP_STORED
            zinfo.external_attr = 0o644 << 16
            zinfo.create_system = 3  # fixed (Unix) so archives don't vary by host OS
            zf.writestr(zinfo, buf.getvalue())


def write_step_npz(out_dir, derived: DerivedCapture) -> dict:
    """Write one deterministic compressed NPZ and return its file record.

    Augments every tensor record with ZIP-derived byte accounting so the
    artifact can be ranked by compressed storage cost across all
    layers/steps/phases. Byte field definitions:
      raw_bytes        ndarray.nbytes (uncompressed in-memory)
      npy_bytes        ZipInfo.file_size (.npy payload incl. numpy header)
      compressed_bytes ZipInfo.compress_size (deflated .npy payload)
      compression_ratio raw_bytes / compressed_bytes (None if compressed==0)
    File-level:
      raw_bytes                sum of tensor raw_bytes (incl. token_ids)
      npy_payload_bytes        sum of tensor npy_bytes
      compressed_payload_bytes sum of tensor compressed_bytes
      archive_bytes            path.stat().st_size (whole .npz on disk)
      zip_overhead_bytes       archive_bytes - compressed_payload_bytes
      compressed_bytes         == archive_bytes (whole .npz on disk; kept for
                               compatibility, defined unambiguously here)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / derived.filename
    _write_npz_deterministic(path, derived.arrays)

    with zipfile.ZipFile(path, "r") as zf:
        info = {zi.filename: zi for zi in zf.infolist()}
    npy_payload_bytes = 0
    compressed_payload_bytes = 0
    for rec in derived.tensor_records:
        zi = info[f"{rec['key']}.npy"]
        npy_bytes = int(zi.file_size)
        comp_bytes = int(zi.compress_size)
        rec["npy_bytes"] = npy_bytes
        rec["compressed_bytes"] = comp_bytes
        rec["compression_ratio"] = (
            rec["raw_bytes"] / comp_bytes if comp_bytes else None
        )
        npy_payload_bytes += npy_bytes
        compressed_payload_bytes += comp_bytes

    archive_bytes = int(path.stat().st_size)
    return {
        "filename": derived.filename,
        "step": derived.step,
        "phase": derived.phase,
        "sequence_length": derived.sequence_length,
        "token_ids_sha256": derived.token_sha256,
        "num_token_ids": len(derived.token_ids),
        "tensors": derived.tensor_records,
        "raw_bytes": int(derived.raw_bytes),
        "npy_payload_bytes": npy_payload_bytes,
        "compressed_payload_bytes": compressed_payload_bytes,
        "archive_bytes": archive_bytes,
        "zip_overhead_bytes": archive_bytes - compressed_payload_bytes,
        "compressed_bytes": archive_bytes,
    }


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def finalize_artifact_dir(out_dir, file_records: list, run_meta: dict,
                          expected_files: int = 12) -> dict:
    """Write storage_report.json, manifest.json, SHA256SUMS in acyclic order."""
    out_dir = Path(out_dir)
    if len(file_records) != expected_files:
        raise ObserverError(
            f"expected {expected_files} capture files, have {len(file_records)}"
        )
    file_records = sorted(file_records, key=lambda r: r["filename"])

    # (b) storage report from NPZ metadata only.  Per-file byte accounting
    #     plus a per-tensor breakdown and a per-quantity aggregate so any
    #     quantity can be ranked by its compressed storage cost across all
    #     layers / steps / phases.
    by_quantity: dict = {}
    for r in file_records:
        for t in r["tensors"]:
            agg = by_quantity.setdefault(
                t["name"],
                {"name": t["name"], "count": 0,
                 "raw_bytes": 0, "npy_bytes": 0, "compressed_bytes": 0},
            )
            agg["count"] += 1
            agg["raw_bytes"] += t["raw_bytes"]
            agg["npy_bytes"] += t.get("npy_bytes", 0)
            agg["compressed_bytes"] += t.get("compressed_bytes", 0)
    ranked_quantities = sorted(
        by_quantity.values(), key=lambda a: a["compressed_bytes"], reverse=True
    )

    storage_report = {
        "files": [
            {
                "filename": r["filename"],
                "raw_bytes": r["raw_bytes"],
                "npy_payload_bytes": r["npy_payload_bytes"],
                "compressed_payload_bytes": r["compressed_payload_bytes"],
                "archive_bytes": r["archive_bytes"],
                "zip_overhead_bytes": r["zip_overhead_bytes"],
                "compressed_bytes": r["compressed_bytes"],
                "tensors": [
                    {
                        "key": t["key"],
                        "name": t["name"],
                        "layer": t["layer"],
                        "raw_bytes": t["raw_bytes"],
                        "npy_bytes": t.get("npy_bytes"),
                        "compressed_bytes": t.get("compressed_bytes"),
                        "compression_ratio": t.get("compression_ratio"),
                    }
                    for t in r["tensors"]
                ],
            }
            for r in file_records
        ],
        "by_quantity": ranked_quantities,
        "total_raw_bytes": sum(r["raw_bytes"] for r in file_records),
        "total_npy_payload_bytes": sum(r["npy_payload_bytes"] for r in file_records),
        "total_compressed_payload_bytes": sum(
            r["compressed_payload_bytes"] for r in file_records
        ),
        "total_archive_bytes": sum(r["archive_bytes"] for r in file_records),
        "total_compressed_bytes": sum(r["compressed_bytes"] for r in file_records),
    }
    storage_path = out_dir / "storage_report.json"
    _dump_json(storage_path, storage_report)

    # (c) hash NPZs and the storage report
    hashes = {r["filename"]: sha256_file(out_dir / r["filename"]) for r in file_records}
    hashes["storage_report.json"] = sha256_file(storage_path)

    # (d) manifest embeds those hashes; never its own
    manifest = {
        "observer_schema_version": OBSERVER_SCHEMA_VERSION,
        "observer_family": OBSERVER_FAMILY,
        "capture_window": sorted(
            {r["step"] for r in file_records}
        ),
        "phases": list(PHASES),
        "finite_assertions_passed": True,
        "files": [
            {**{k: v for k, v in r.items()}, "sha256": hashes[r["filename"]]}
            for r in file_records
        ],
        "storage_report_sha256": hashes["storage_report.json"],
        "total_raw_bytes": storage_report["total_raw_bytes"],
        "total_compressed_bytes": storage_report["total_compressed_bytes"],
        "total_compressed_payload_bytes": storage_report["total_compressed_payload_bytes"],
        **run_meta,
    }
    manifest_path = out_dir / "manifest.json"
    _dump_json(manifest_path, manifest)

    # (e) SHA256SUMS covers NPZs + storage_report.json + manifest.json
    hashes["manifest.json"] = sha256_file(manifest_path)
    lines = [f"{hashes[name]}  {name}" for name in sorted(hashes)]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    return manifest
