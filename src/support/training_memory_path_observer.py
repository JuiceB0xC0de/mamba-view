"""Memory Path recorder: token-by-token replay of the OFFICIAL decode cache.

WHAT THIS IS
  A reusable, storage-independent recorder that replays one already-tokenized
  sequence through a loaded model token by token, under
  ``torch.inference_mode()``, using the model's OWN inference cache and step
  path, and streams a before/after snapshot of every layer's decode cache
  (angle_state, ssm_state, k_state, v_state) to a caller-supplied sink for
  every (layer, token) pair. It is an instrument. It does not train, does not
  mutate the model, and does not decide parity.

WHAT THE PAYLOADS MEAN
  Everything captured here is a MODEL-OWNED CACHE MEASUREMENT: the contents of
  the tensors the deployed step path actually carries. Whether those cache
  tensors equal the causal state trajectory of the source-derived reference
  recurrence is decided exclusively by analysis/memory_path_parity.py. Until a
  complete PASS parity report for this exact model/revision is presented via
  ``enable_state_claims``, every payload is labeled
  claim_level="cache-measurement" and state-trajectory claims are prohibited.
  Reference-derived pathway tensors (earlier read, diagonal, feedthrough,
  gamma/deferred write halves) come from the reference observer, not from this
  replay; ``reference_payload_to_sink_payload`` adapts them to the same sink
  interface and labels them claim_level="reference-derived".

  A norm summary is a summary. It is never the state itself, and SummarySink
  output must not be quoted as a state trajectory.

DISCIPLINE
  * RNG (Python, Torch CPU, every CUDA device) is snapshotted before replay
    and restored in a ``finally`` block, including on sink exceptions.
  * Parameters, gradients, optimizer state, and train/eval mode are never
    modified; replay runs under torch.inference_mode().
  * Each independent sequence gets a fresh zero-initialized cache.
  * Every retained tensor is a detached, compact clone (no view aliasing of
    live cache buffers, no autograd graphs).
  * Non-finite values in outputs or caches fail loudly.
  * Layer count and geometry are discovered from the model at runtime; nothing
    hardcodes 12 layers.
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_ANALYSIS = _ROOT / "analysis"
for _p in (str(_ROOT), str(_ANALYSIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import memory_path_parity as mpp  # noqa: E402

SCHEMA_VERSION = "memory-path-recorder-v1"

CACHE_FIELDS = ("angle_state", "ssm_state", "k_state", "v_state")

# reference-observer tensor fields adapted by reference_payload_to_sink_payload
REFERENCE_TENSOR_FIELDS = (
    "state_read", "gamma_write", "deferred_write", "earlier_read", "diagonal",
    "feedthrough", "pending_k", "pending_v", "K_trap", "C_query", "phase",
)


class MemoryPathError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# RNG discipline (same shape as training_time_observer.rng_snapshot/restore,
# restated here so this module does not import training code)
# ---------------------------------------------------------------------------


def rng_snapshot() -> dict:
    snap = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state_all()
    return snap


def rng_restore(snap: dict) -> None:
    random.setstate(snap["python"])
    torch.set_rng_state(snap["torch_cpu"])
    if "torch_cuda" in snap and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap["torch_cuda"])


# ---------------------------------------------------------------------------
# geometry discovery (runtime; no hardcoded layer count)
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    n_layers: int
    nheads: int
    headdim: int
    d_state: int
    n_rope_angles: int
    mimo_rank: int
    d_model: int


_MIXER_ATTRS = ("allocate_inference_cache", "step", "nheads", "headdim",
                "d_state", "num_rope_angles", "mimo_rank")


def discover_mixers(model) -> list:
    """Every decode-capable mixer, in registration order."""
    found = [(name, mod) for name, mod in model.named_modules()
             if all(hasattr(mod, a) for a in _MIXER_ATTRS)]
    if not found:
        raise MemoryPathError(
            f"no decode-capable mixers found: no module exposes {_MIXER_ATTRS}")
    return found


def discover_geometry(model) -> Geometry:
    mixers = discover_mixers(model)
    _, m0 = mixers[0]
    geo = Geometry(
        n_layers=len(mixers), nheads=int(m0.nheads), headdim=int(m0.headdim),
        d_state=int(m0.d_state), n_rope_angles=int(m0.num_rope_angles),
        mimo_rank=int(m0.mimo_rank),
        d_model=int(m0.in_proj.weight.shape[1]) if hasattr(m0, "in_proj")
        else -1,
    )
    for name, m in mixers[1:]:
        for attr in ("nheads", "headdim", "d_state", "num_rope_angles",
                     "mimo_rank"):
            if int(getattr(m, attr)) != getattr(
                    geo, {"num_rope_angles": "n_rope_angles"}.get(attr, attr)):
                raise MemoryPathError(
                    f"heterogeneous geometry at {name}.{attr}; the recorder "
                    "does not support mixed-geometry stacks")
    return geo


def expected_cache_shapes(geo: Geometry, batch: int = 1) -> dict:
    """The OFFICIAL decode cache shapes (mamba3.py step docstring L322-334)."""
    return {
        "angle_state": (batch, geo.nheads, geo.n_rope_angles),
        "ssm_state": (batch, geo.nheads, geo.headdim, geo.d_state),
        "k_state": (batch, geo.mimo_rank, geo.nheads, geo.d_state),
        "v_state": (batch, geo.nheads, geo.headdim),
    }


def assert_cache_shapes(cache: dict, geo: Geometry, where: str = "") -> None:
    want = expected_cache_shapes(geo)
    for k in CACHE_FIELDS:
        got = tuple(cache[k].shape)
        if got != want[k]:
            raise MemoryPathError(
                f"official cache shape violated{(' at ' + where) if where else ''}: "
                f"{k} is {got}, the pinned convention is {want[k]}")


# ---------------------------------------------------------------------------
# inference-params shim (the mamba3 module reads only these two attributes)
# ---------------------------------------------------------------------------


@dataclass
class SimpleInferenceParams:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# snapshotting
# ---------------------------------------------------------------------------


def _compact_clone(t: torch.Tensor) -> torch.Tensor:
    """Detached, contiguous, freshly-allocated copy; never a view."""
    out = t.detach().clone().contiguous()
    if out.requires_grad or out.grad_fn is not None:
        raise MemoryPathError("snapshot retained an autograd graph")
    return out


def _check_finite(t: torch.Tensor, what: str) -> None:
    if not torch.isfinite(t).all():
        raise MemoryPathError(f"non-finite values in {what}; refusing to record")


def snapshot_layer_cache(cache_tuple) -> dict:
    angle, ssm, k, v = cache_tuple
    return {"angle_state": _compact_clone(angle), "ssm_state": _compact_clone(ssm),
            "k_state": _compact_clone(k), "v_state": _compact_clone(v)}


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------


class FullStateSink:
    """Retains exact tensors. For tiny fixtures and selected raw calibration
    slices ONLY: at the real geometry this costs gigabytes per sequence (use
    ``estimate_storage_bytes``). Byte accounting is exact."""

    def __init__(self):
        self.sequences = []
        self._current = None
        self.bytes_retained = 0

    def begin_sequence(self, metadata: dict) -> None:
        self._current = {"metadata": dict(metadata), "events": []}

    def observe_token(self, layer: int, token_index: int, payload: dict) -> None:
        if self._current is None:
            raise MemoryPathError("observe_token before begin_sequence")
        kept = {}
        for k, v in payload.items():
            if torch.is_tensor(v):
                kept[k] = v
                self.bytes_retained += v.numel() * v.element_size()
            else:
                kept[k] = v
        self._current["events"].append(
            {"layer": layer, "token": token_index, "payload": kept})

    def end_sequence(self, metadata: dict) -> None:
        self._current["end_metadata"] = dict(metadata)
        self.sequences.append(self._current)
        self._current = None


# Per-head scalar summaries. A row costs O(H) per (layer, token): it never
# scales with headdim * d_state. THE SUMMARY IS NOT THE STATE.
_SUMMARY_NORM_FIELDS = {
    "state_before_norm": "state_before",
    "state_after_norm": "state_after",
    "resolved_write_norm": "gamma_write",
    "pending_k_norm": "pending_k",
    "pending_v_norm": "pending_v",
    "earlier_read_norm": "earlier_read",
    "diagonal_norm": "diagonal",
    "feedthrough_norm": "feedthrough",
    "pre_gate_norm": "pre_gate",
    "post_gate_norm": "post_gate",
}


def _per_head_norm(t: torch.Tensor) -> torch.Tensor:
    """(H, ...) -> (H,) L2 norm over everything but the head axis."""
    return t.reshape(t.shape[0], -1).norm(dim=-1)


class SummarySink:
    """Constant-memory-per-event streaming summaries: per layer/head/token
    scalars only. Retained size grows with n_layers * T * H, NEVER with
    headdim * d_state. Effective rank is computed only when explicitly
    requested for selected (layer, token) pairs (it needs an SVD of the full
    state, so it is opt-in by coordinate, not a default)."""

    def __init__(self, effective_rank_coords: set | None = None):
        self.rows = []           # list of dicts of floats / (H,) float tensors
        self.effective_rank_coords = set(effective_rank_coords or ())
        self._meta = None

    def begin_sequence(self, metadata: dict) -> None:
        self._meta = dict(metadata)

    def observe_token(self, layer: int, token_index: int, payload: dict) -> None:
        row = {"layer": layer, "token": token_index,
               "claim_level": payload.get("claim_level")}
        for out_name, in_name in _SUMMARY_NORM_FIELDS.items():
            v = payload.get(in_name)
            if torch.is_tensor(v):
                row[out_name] = _per_head_norm(v.float()).clone()

        sb, sa = payload.get("state_before"), payload.get("state_after")
        if torch.is_tensor(sb) and torch.is_tensor(sa):
            d = (sa.float() - sb.float())
            row["state_delta_norm"] = _per_head_norm(d).clone()
            nb = _per_head_norm(sb.float())
            na = _per_head_norm(sa.float())
            dot = (sb.float().reshape(sb.shape[0], -1)
                   * sa.float().reshape(sa.shape[0], -1)).sum(-1)
            denom = nb * na
            # cosine only where defined (both norms nonzero)
            cos = torch.where(denom > 0, dot / denom.clamp_min(1e-30),
                              torch.full_like(dot, float("nan")))
            row["state_cosine_before_after"] = cos.clone()
            if (layer, token_index) in self.effective_rank_coords:
                row["effective_rank"] = self._effective_rank(sa.float())

        g = payload.get("gate_factor")
        if torch.is_tensor(g):
            gf = g.float()
            row["gate_factor_mean"] = float(gf.mean())
            row["gate_factor_min"] = float(gf.min())
            row["gate_factor_max"] = float(gf.max())

        o = payload.get("mixer_output")
        if torch.is_tensor(o):
            row["mixer_output_norm"] = float(o.float().norm())

        for k, v in row.items():
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                if k != "state_cosine_before_after":  # NaN declared where undefined
                    raise MemoryPathError(f"non-finite summary {k}")
        self.rows.append(row)

    @staticmethod
    def _effective_rank(state: torch.Tensor) -> torch.Tensor:
        """(H, P, N) -> (H,) exp(entropy of normalized singular values)."""
        s = torch.linalg.svdvals(state.reshape(state.shape[0],
                                               state.shape[-2], -1)
                                 if state.dim() > 2 else state.unsqueeze(0))
        p = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        ent = -(p * p.clamp_min(1e-30).log()).sum(dim=-1)
        return ent.exp()

    def end_sequence(self, metadata: dict) -> None:
        self._meta = None

    def retained_bytes(self) -> int:
        total = 0
        for row in self.rows:
            for v in row.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
                elif isinstance(v, float):
                    total += 8
        return total


# ---------------------------------------------------------------------------
# storage estimator (exact arithmetic; verified by tests at real geometry)
# ---------------------------------------------------------------------------


def estimate_storage_bytes(T: int, L: int, H: int, P: int, N: int, K: int,
                           R: int = 1, dtype_bytes: int = 4,
                           n_summary_fields: int = 13) -> dict:
    """Exact byte counts for one sequence of T tokens across L layers.

    ssm before+after:  2 * T * L * H * P * N
    angle before+after: 2 * T * L * H * K
    k before+after:     2 * T * L * R * H * N
    v before+after:     2 * T * L * H * P
    pathway tensors (reference observer, per token per layer):
      gamma_write + deferred_write: 2 * H * P * N
      earlier/diagonal/feedthrough/pending_v: 4 * H * R * P
      pending_k + K_trap + C_query: 3 * H * R * N
      phase: H * K
    summary-only: n_summary_fields per-head scalars per (layer, token).
    """
    b = dtype_bytes
    ssm = 2 * T * L * H * P * N * b
    angle = 2 * T * L * H * K * b
    k = 2 * T * L * R * H * N * b
    v = 2 * T * L * H * P * b
    pathway = T * L * b * (2 * H * P * N + 4 * H * R * P
                           + 3 * H * R * N + H * K)
    summary = T * L * H * n_summary_fields * b
    return {
        "ssm_state_before_after": ssm,
        "angle_state_before_after": angle,
        "k_state_before_after": k,
        "v_state_before_after": v,
        "cache_total": ssm + angle + k + v,
        "pathway_tensors": pathway,
        "full_everything": ssm + angle + k + v + pathway,
        "summary_only": summary,
        "params": {"T": T, "L": L, "H": H, "P": P, "N": N, "K": K, "R": R,
                   "dtype_bytes": b, "n_summary_fields": n_summary_fields},
    }


# ---------------------------------------------------------------------------
# reference-observer adapter
# ---------------------------------------------------------------------------


def reference_payload_to_sink_payload(ref_payload: dict) -> dict:
    """Adapt a reference_block_forward observer payload to the sink payload
    vocabulary. Everything here is REFERENCE-DERIVED, not a model-owned cache
    measurement, and is labeled as such. prefill_accum is future-aware and is
    deliberately NOT mapped to any state field."""
    out = {"claim_level": "reference-derived"}
    for k in REFERENCE_TENSOR_FIELDS:
        if k in ref_payload:
            out[k] = ref_payload[k]
    if "state_read" in ref_payload:
        # the reference CAUSAL read-state is the closest analogue of
        # "state before the token acts"; named explicitly, not renamed to the
        # official cache convention (see memory_path_parity.STATE_CONVENTION)
        out["state_before"] = ref_payload["state_read"]
    for k in ("write_source_token", "gamma_write_resolution_token",
              "deferred_write_resolution_token"):
        if k in ref_payload:
            out[k] = ref_payload[k]
    return out


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------


class MemoryPathRecorder:
    """Replays one tokenized sequence through the official step path and
    streams per-(layer, token) cache snapshots to a sink."""

    def __init__(self, model):
        self.model = model
        self.mixers = discover_mixers(model)
        self.geometry = discover_geometry(model)
        self.state_claims_enabled = False
        self._state_claims_reason = ("no parity report presented; state "
                                     "claims prohibited")

    # -- state-claim gating -------------------------------------------------

    def enable_state_claims(self, parity_report: dict,
                            expected_source_hashes: dict | None = None) -> None:
        """Enable state-claim labeling. ONLY a complete PASS report for the
        pinned model/revision is accepted; everything else raises."""
        ok, reason = mpp.validate_report_for_state_claims(
            parity_report, expected_source_hashes=expected_source_hashes)
        if not ok:
            self.state_claims_enabled = False
            self._state_claims_reason = reason
            raise MemoryPathError(f"state-claim mode refused: {reason}")
        self.state_claims_enabled = True
        self._state_claims_reason = reason

    @property
    def claim_level(self) -> str:
        return ("state-trajectory" if self.state_claims_enabled
                else "cache-measurement")

    # -- replay ---------------------------------------------------------------

    def _fresh_inference_params(self, max_seqlen: int) -> SimpleInferenceParams:
        ip = SimpleInferenceParams(max_seqlen=max_seqlen, max_batch_size=1)
        for li, (_, mixer) in enumerate(self.mixers):
            cache = mixer.allocate_inference_cache(1, max_seqlen)
            for t in cache:
                t.zero_()
            ip.key_value_memory_dict[li] = cache
        return ip

    def record_sequence(self, token_ids, sink, metadata: dict | None = None):
        """Replay token_ids (1D iterable of ints) token by token.

        Calls sink.begin_sequence, then sink.observe_token(layer, t, payload)
        for every layer at every token, then sink.end_sequence. Payload keys:
        state_before/state_after (ssm), angle_before/after, k_before/after,
        v_before/after, mixer_output is NOT captured per layer here (the step
        path returns only the block output); the model-level output for the
        token is passed in the *_sequence metadata stream instead.
        """
        ids = torch.as_tensor(list(token_ids), dtype=torch.int64)
        if ids.dim() != 1 or ids.numel() == 0:
            raise MemoryPathError("token_ids must be a non-empty 1D sequence")
        T = int(ids.numel())
        device = next(self.model.parameters()).device

        was_training = self.model.training
        param_versions = [p._version for p in self.model.parameters()]
        snap = rng_snapshot()
        meta = dict(metadata or {})
        meta.update({"schema_version": SCHEMA_VERSION,
                     "claim_level": self.claim_level,
                     "claim_reason": self._state_claims_reason,
                     "n_tokens": T,
                     "geometry": vars(self.geometry)})
        try:
            with torch.inference_mode():
                ip = self._fresh_inference_params(T)
                # seqlen_offset > 0 routes mamba3.forward to the step path
                # from the very first token (mamba3.py L171-173); caches are
                # explicitly zeroed above, so token 0 starts from zero state.
                ip.seqlen_offset = 1
                sink.begin_sequence(meta)
                # Finiteness is accumulated on device and resolved once at the
                # end of the sequence.  Checking inline forced a host sync per
                # field per layer per token, which serialized the replay
                # against the GPU roughly 15k times per sequence.  The same
                # values are checked and the same layer/token is reported.
                finite_labels = []
                finite_flags = []
                _prof = {"before": 0, "forward": 0, "after": 0, "observe": 0}
                for t in range(T):
                    _m = time.perf_counter_ns()
                    before = {li: snapshot_layer_cache(ip.key_value_memory_dict[li])
                              for li in range(self.geometry.n_layers)}
                    _prof["before"] += time.perf_counter_ns() - _m
                    _m = time.perf_counter_ns()
                    out = self.model(ids[t:t + 1].unsqueeze(0).to(device),
                                     inference_params=ip)
                    _prof["forward"] += time.perf_counter_ns() - _m
                    out_t = getattr(out, "logits", out)
                    if torch.is_tensor(out_t):
                        finite_labels.append(f"model output at token {t}")
                        finite_flags.append(torch.isfinite(out_t).all())
                    for li in range(self.geometry.n_layers):
                        _m = time.perf_counter_ns()
                        after = snapshot_layer_cache(ip.key_value_memory_dict[li])
                        _prof["after"] += time.perf_counter_ns() - _m
                        assert_cache_shapes(after, self.geometry,
                                            f"layer {li} token {t}")
                        for k in CACHE_FIELDS:
                            finite_labels.append(f"{k} layer {li} token {t}")
                            finite_flags.append(torch.isfinite(after[k]).all())
                        payload = {
                            "claim_level": self.claim_level,
                            "state_before": before[li]["ssm_state"][0],
                            "state_after": after["ssm_state"][0],
                            "angle_before": before[li]["angle_state"][0],
                            "angle_after": after["angle_state"][0],
                            "k_before": before[li]["k_state"][0],
                            "k_after": after["k_state"][0],
                            "v_before": before[li]["v_state"][0],
                            "v_after": after["v_state"][0],
                        }
                        _m = time.perf_counter_ns()
                        sink.observe_token(li, t, payload)
                        _prof["observe"] += time.perf_counter_ns() - _m
                    ip.seqlen_offset += 1
                print(
                    "replay profile "
                    + " ".join(f"{k}={v / 1e9:.2f}s" for k, v in _prof.items())
                    + f" tokens={T}",
                    flush=True,
                )
                if finite_flags:
                    resolved = torch.stack(finite_flags).cpu()
                    bad = (~resolved).nonzero()
                    if bad.numel():
                        raise MemoryPathError(
                            "non-finite values in "
                            f"{finite_labels[int(bad[0])]}; refusing to record"
                        )
                sink.end_sequence(meta)
        finally:
            rng_restore(snap)
            # replay must be side-effect-free on the model
            if self.model.training != was_training:
                raise MemoryPathError("replay changed model train/eval mode")
            for p, ver in zip(self.model.parameters(), param_versions):
                if p._version != ver:
                    raise MemoryPathError("replay mutated a model parameter")
        return meta
