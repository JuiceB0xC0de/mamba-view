"""Training memory-path camera for Mamba-3 (Camera 4, steps 20-25).

Modal-free by design so every mechanism is CPU-testable. The Modal trainer
(modal_memory_path_camera.py) imports this module; nothing here imports modal,
wandb, or CUDA-only code.

Camera 4 asks how each token is written, retained, read, gated and delivered to
the mixer output during the same mundane optimizer updates 20-25.

Three evidence classes that never share a label
-----------------------------------------------
1. ``live-prefill`` -- actual mixer inputs and outputs from the real training
   forward, and from the identical-example post-update replay.
2. ``cache-measurement`` -- summaries of the model-owned ``angle_state``,
   ``ssm_state``, ``k_state`` and ``v_state`` during an RNG-preserved
   token-by-token replay through the deployed step path.
3. ``reference-derived`` -- the source-derived sufficient basis and
   write/read/output decomposition from ``reference_block_forward``.

Every stored array carries an evidence-class code. One key is never reused
across two classes.

The parked state boundary
-------------------------
Parity run ``umv1n6he`` ended ``MEASURED_NEEDS_REVIEW``, NOT ``PASS``.
Therefore ``STATE_CLAIMS_ENABLED = False`` and this camera prohibits ``h_t``,
causal-state, retention-duration and kernel-state-equivalence claims. Official
caches may be described only as cache magnitude/direction summaries; no prose
may say a token is "stored" or "retained for" a duration on their basis.

Camera 4 records the path without pretending the parked boundary has
disappeared. Enabling state claims requires a complete PASS report, which the
foundation's own gate refuses to synthesize.

Read-only with respect to the optimization process
--------------------------------------------------
The camera never calls ``backward()``, ``clip_grad_norm_()`` or
``optimizer.step()``. Its mixer hooks take detached compact clones and return
``None``. Its replays run under ``torch.no_grad()`` / inference mode with RNG
snapshotted and restored, and assert afterwards that RNG state, training mode,
parameter versions, gradients (including ``None`` identity) and optimizer state
are all unchanged.

Foundation reuse
----------------
``MemoryPathRecorder``, geometry discovery, RNG discipline, cache-shape checks
and sink semantics come from ``training_memory_path_observer``.
``reference_block_forward`` comes from ``analysis.reference_recurrence``.
``extract_layer_params``, ``resolve_arm`` and the output parity
metrics/tolerances come from ``analysis.full_model_smoke``. None of those files
is edited or transcribed here: where a helper lacked a field, it is wrapped or
subclassed in this module instead.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Camera 1 first: it registers the canonical `mamba3_core` module identity that
# every analysis import below inherits.
import training_time_observer as tto  # noqa: E402
import training_memory_path_observer as tmpo  # noqa: E402

from analysis import full_model_smoke as fms  # noqa: E402
from analysis import reference_recurrence as refrec  # noqa: E402
from analysis import mamba3_core as m3c  # noqa: E402

MemoryPathError = tmpo.MemoryPathError
ObserverError = tto.ObserverError
NonFiniteCaptureError = tto.NonFiniteCaptureError
GeometryMismatchError = tto.GeometryMismatchError

write_step_npz = tto.write_step_npz
sha256_file = tto.sha256_file
token_ids_sha256 = tto.token_ids_sha256

MEMORY_PATH_CAMERA_SCHEMA_VERSION = 1
CAMERA_NUMBER = 4
CAMERA_FAMILY = "memory_path"
CAPTURE_START_STEP = 20
CAPTURE_END_STEP = 25
CAPTURE_WINDOW = frozenset(range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1))
PHASES = ("pre_update", "post_update")

# The parked G3 boundary. Not a default to be overridden by a caller: the
# foundation's own gate refuses anything short of a complete PASS report.
STATE_CLAIMS_ENABLED = False
PARITY_RUN_ID = "umv1n6he"
PARITY_VERDICT = "MEASURED_NEEDS_REVIEW"
STATE_CLAIMS_REASON = (
    f"parity run {PARITY_RUN_ID} ended {PARITY_VERDICT}, not PASS; h_t, "
    "causal-state, retention-duration and kernel-state-equivalence claims are "
    "prohibited in v1"
)

# Evidence classes. Stored under distinct key prefixes so a live value, an
# official-cache summary and a reference-derived quantity can never collide.
EVIDENCE_LIVE = "live-prefill"
EVIDENCE_CACHE = "cache-measurement"
EVIDENCE_REFERENCE = "reference-derived"
EVIDENCE_CLASSES = (EVIDENCE_LIVE, EVIDENCE_CACHE, EVIDENCE_REFERENCE)
EVIDENCE_PREFIX = {
    EVIDENCE_LIVE: "live",
    EVIDENCE_CACHE: "cache",
    EVIDENCE_REFERENCE: "ref",
}
EVIDENCE_CODE = {name: i for i, name in enumerate(EVIDENCE_CLASSES)}

CACHE_FIELDS = tmpo.CACHE_FIELDS  # angle_state, ssm_state, k_state, v_state

# Which axis of each official cache tensor (batch already indexed away) is the
# head axis. k_state is (mimo_rank, nheads, d_state) -- rank first -- so a naive
# "axis 0 is heads" reduction would silently summarize the wrong axis.
_CACHE_HEAD_AXIS = {
    "angle_state": 0,   # (H, K)
    "ssm_state": 0,     # (H, P, N)
    "k_state": 1,       # (R, H, N)
    "v_state": 0,       # (H, P)
}
_CACHE_PAYLOAD_KEYS = {
    "ssm_state": ("state_before", "state_after"),
    "angle_state": ("angle_before", "angle_after"),
    "k_state": ("k_before", "k_after"),
    "v_state": ("v_before", "v_after"),
}
CACHE_SUMMARY_FIELDS = ("before_norm", "after_norm", "delta_norm",
                        "cosine", "cosine_valid")

# Reference-derived per-token scalar controls, all [T, H].
REF_SCALAR_FIELDS = ("Delta", "A", "ADT", "alpha", "lambda", "gamma",
                     "shifted_gamma", "trap_scale")
# Reference-derived read pathways, all [T, H, R, P].
REF_PATHWAYS = ("earlier", "diagonal", "feedthrough")

# fp32 gate for the internal algebraic identities this camera asserts. Matches
# the reference module's own internal validate= tolerance; these are exact
# algebraic rearrangements of fp32 tensors, not physical comparisons.
CLOSURE_ATOL = 1e-5
CLOSURE_RTOL = 1e-5
# gamma_n for the out-projection contraction, set from the REALIZED d_inner at
# construction time (see TrainingMemoryPathCamera.__init__). Never a literal.
CLOSURE_GAMMA = None


FLOAT32_UNIT_ROUNDOFF = 5.9604644775390625e-08   # eps/2 for IEEE binary32


def closure_deviation(total: torch.Tensor, reference: torch.Tensor,
                      abs_products) -> tuple:
    """Check a sum-of-projected-terms identity against its own error bound.

    The identity is exact in exact arithmetic; the observed difference is
    float32 rounding introduced by an ASSOCIATIVITY CHANGE. The reference
    computes one matmul of the summed pre-gate tensor; this decomposition
    computes three separate matmuls and adds them. Both contract over
    ``d_inner`` (1536 on the pinned model), and reassociating a 1536-term dot
    product changes its rounding.

    The bound is therefore the textbook one for matrix multiplication,

        |fl(AB) - AB| <= gamma_n * |A| @ |B|,   gamma_n ~= n * u

    so the denominator must be the CONTRACTED PRODUCT magnitudes |A| @ |B|,
    not the output magnitudes |A @ B|. Using the output misses the
    cancellation that happens inside the contraction: at d_inner = 1536,
    gamma_n ~= 9.2e-5, and a gap of that size is the bound being met, not an
    algebra error.

    ``abs_products`` supplies |A| @ |B| per pathway. n is taken from the
    realized contraction length, so nothing here is fitted to an observation:
    a genuine decomposition error is O(1) relative and still fails.

    Returns (within_bound, max_relative_deviation, max_absolute_gap).
    """
    gap = (total - reference).abs()
    scale = torch.zeros_like(gap)
    for product in abs_products:
        scale = scale + product
    if CLOSURE_GAMMA is None:
        raise MemoryPathError(
            "closure gamma_n was never set from the realized contraction "
            "length; refusing to check against an unspecified bound"
        )
    tolerance = CLOSURE_ATOL + scale * CLOSURE_GAMMA
    positive = scale > 0
    relative = torch.zeros_like(gap)
    relative[positive] = gap[positive] / scale[positive]
    within = bool((gap <= tolerance).all())
    return within, float(relative.max()), float(gap.max())

# ---------------------------------------------------------------------------
# decode shape adapter (ADDENDUM-camera4-mamba3-decode-shape-adapter-v1.md)
#
# The pinned wheel's Mamba3.forward() hands step() a (B, 1, D) tensor while
# step() requires (B, D). The parity campaign already proved the 2-D step path
# imports, executes and reaches the official kernel on A10G, so the defect is
# the singleton sequence-axis contract at the forward->step API boundary, not
# recurrence math or kernel availability.
#
# This adapter squeezes that singleton axis on the way in and restores it on the
# way out. It changes RANK PLUMBING ONLY: no recurrence equation, parameter,
# cache or kernel is touched. It is installed exclusively around the official
# cache replay and never exists during the live training forward, loss,
# backward, clipping, optimizer step, reference derivation, or off mode.
# ---------------------------------------------------------------------------

DECODE_SHAPE_ADAPTER_VERSION = 1
DECODE_SHAPE_ADAPTER_SCOPE = "official_cache_replay_only"
DECODE_SHAPE_ADAPTER_INPUT = "(B,1,D)"
DECODE_SHAPE_ADAPTER_ORIGINAL_STEP_INPUT = "(B,D)"
DECODE_SHAPE_ADAPTER_OUTPUT_RESTORED = "(B,1,D)"
UPSTREAM_ISSUE_RECORD = "UPSTREAM-ISSUE-mamba3-step-shape.md"
UPSTREAM_ISSUE_URL = "https://github.com/state-spaces/mamba/issues/1014"

OFFICIAL_CACHE_EVIDENCE_LABEL = (
    "model-owned cache measurements produced by the pinned official "
    "Mamba3.step recurrence/kernel under a local singleton-axis compatibility "
    "adapter at the forward->step API boundary; NOT the unmodified upstream "
    "decode path"
)

_ADAPTER_MARK = "_camera4_decode_shape_adapter"


class DecodeShapeAdapterError(MemoryPathError):
    """The decode shape adapter refused an input or a result."""


def _make_adapted_step(original, label: str):
    """Wrap one bound ``step`` with the singleton-axis adapter."""

    def adapted_step(u, *args, **kwargs):
        if not torch.is_tensor(u):
            raise DecodeShapeAdapterError(
                f"{label}: decode step received {type(u).__name__}, not a tensor"
            )
        if u.dim() == 2:
            # Direct 2-D API, exactly as memory_path_parity.py uses it:
            # delegate untouched and return the result untouched.
            return original(u, *args, **kwargs)
        if u.dim() != 3:
            raise DecodeShapeAdapterError(
                f"{label}: decode step expects (B,D) or (B,1,D), got rank "
                f"{u.dim()} {tuple(u.shape)}"
            )
        if int(u.shape[1]) != 1:
            raise DecodeShapeAdapterError(
                f"{label}: decode step received sequence length "
                f"{int(u.shape[1])}; the adapter only restores a SINGLETON "
                "axis and refuses to guess a multi-token contract"
            )
        result = original(u.squeeze(1), *args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 5:
            raise DecodeShapeAdapterError(
                f"{label}: official step returned "
                f"{len(result) if isinstance(result, tuple) else type(result).__name__}"
                " items, expected 5"
            )
        out = result[0]
        if not torch.is_tensor(out) or out.dim() != 2:
            raise DecodeShapeAdapterError(
                f"{label}: official step output is not 2-D; the adapter will "
                "not restore an axis onto an unexpected shape"
            )
        # Only the model output regains the axis. The four cache/state objects
        # are returned exactly as the official step produced them.
        return (out.unsqueeze(1),) + tuple(result[1:])

    setattr(adapted_step, _ADAPTER_MARK, True)
    return adapted_step


@contextmanager
def decode_shape_adapter(mixers):
    """Install the adapter on every mixer for the duration of one replay.

    The exact prior instance state is restored in ``finally``, including when
    the replay or a sink raises: if ``step`` was an instance attribute it is put
    back verbatim, otherwise the temporary attribute is deleted so normal class
    lookup resumes. A saved bound method is never left installed permanently.
    """
    for label, mixer in mixers:
        if getattr(getattr(mixer, "step", None), _ADAPTER_MARK, False):
            raise DecodeShapeAdapterError(
                f"{label}: decode shape adapter is already installed; nested "
                "installation would double-squeeze"
            )
    installed = []
    try:
        for label, mixer in mixers:
            had_own = "step" in mixer.__dict__
            prior = mixer.__dict__.get("step")
            mixer.step = _make_adapted_step(mixer.step, label)
            installed.append((mixer, had_own, prior))
        yield
    finally:
        for mixer, had_own, prior in reversed(installed):
            if had_own:
                mixer.__dict__["step"] = prior
            else:
                mixer.__dict__.pop("step", None)


def adapter_installed(mixers) -> bool:
    return any(
        getattr(getattr(mixer, "step", None), _ADAPTER_MARK, False)
        for _, mixer in mixers
    )


# ---------------------------------------------------------------------------
# official-cache summary sink (all four families)
# ---------------------------------------------------------------------------


def _head_first(field_name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Move the head axis to position 0 so per-head reduction is correct."""
    axis = _CACHE_HEAD_AXIS[field_name]
    return tensor if axis == 0 else tensor.movedim(axis, 0)


def _per_head_norm(tensor: torch.Tensor) -> torch.Tensor:
    """(H, ...) -> (H,) L2 over everything but the head axis."""
    return tensor.reshape(tensor.shape[0], -1).norm(dim=-1)


class TrainingCacheSummarySink:
    """Constant-memory streaming summaries for ALL FOUR official cache families.

    The foundation's ``SummarySink`` computes delta/cosine for the SSM state
    only. Camera 4 needs the same treatment for ``angle_state``, ``k_state``
    and ``v_state``, so this sink extends the semantics rather than editing the
    provenance-bearing foundation file.

    Retains per layer/token/head scalars only -- never a full cache matrix.
    Cosine is stored only where defined (both norms nonzero); elsewhere it is
    NaN with a paired validity mask. Zero would assert orthogonality.
    """

    claim_level = EVIDENCE_CACHE

    def __init__(self):
        self.rows = []
        self._meta = None
        self.sequence_count = 0

    def begin_sequence(self, metadata: dict) -> None:
        self._meta = dict(metadata)
        self.sequence_count += 1

    def observe_token(self, layer: int, token_index: int, payload: dict) -> None:
        declared = payload.get("claim_level")
        if declared != EVIDENCE_CACHE:
            raise MemoryPathError(
                f"official cache payload declared claim_level {declared!r}; "
                f"Camera 4 v1 accepts only {EVIDENCE_CACHE!r} "
                f"({STATE_CLAIMS_REASON})"
            )
        row = {"layer": int(layer), "token": int(token_index),
               "claim_level": EVIDENCE_CACHE}
        for field_name, (before_key, after_key) in _CACHE_PAYLOAD_KEYS.items():
            before = payload.get(before_key)
            after = payload.get(after_key)
            if not (torch.is_tensor(before) and torch.is_tensor(after)):
                raise MemoryPathError(
                    f"official cache payload missing {field_name} at layer "
                    f"{layer} token {token_index}"
                )
            b = _head_first(field_name, before.float())
            a = _head_first(field_name, after.float())
            if b.shape != a.shape:
                raise MemoryPathError(
                    f"{field_name} before/after shape mismatch: "
                    f"{tuple(b.shape)} vs {tuple(a.shape)}"
                )
            nb = _per_head_norm(b)
            na = _per_head_norm(a)
            dot = (b.reshape(b.shape[0], -1) * a.reshape(a.shape[0], -1)).sum(-1)
            denom = nb * na
            valid = denom > 0
            cosine = torch.where(
                valid, dot / denom.clamp_min(1e-30),
                torch.full_like(dot, float("nan")),
            )
            row[f"{field_name}_before_norm"] = nb.clone()
            row[f"{field_name}_after_norm"] = na.clone()
            row[f"{field_name}_delta_norm"] = _per_head_norm(a - b).clone()
            row[f"{field_name}_cosine"] = cosine.clone()
            row[f"{field_name}_cosine_valid"] = valid.to(torch.uint8).clone()
        # Finiteness is asserted in to_arrays, on host, after one batched
        # transfer.  Checking here forced a device sync per field per token,
        # which serialized the replay against the GPU roughly 30k times per
        # sequence.  Same values, same guarantee, same failure.
        self.rows.append(row)

    def end_sequence(self, metadata: dict) -> None:
        self._meta = None

    def retained_bytes(self) -> int:
        total = 0
        for row in self.rows:
            for value in row.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    def to_arrays(self, n_layers: int, n_tokens: int, nheads: int) -> dict:
        """Pack rows into [L, T, H] arrays keyed by family and quantity."""
        expected = n_layers * n_tokens
        if len(self.rows) != expected:
            raise MemoryPathError(
                f"cache summary has {len(self.rows)} rows, expected "
                f"{expected} (= {n_layers} layers x {n_tokens} tokens)"
            )
        # One device-to-host transfer per key instead of one per (row, key).
        # Row order is preserved by stacking, and placement uses the recorded
        # layer/token indices, so the packed arrays are identical to the
        # per-row version.
        layer_index = np.fromiter(
            (row["layer"] for row in self.rows), dtype=np.int64,
            count=len(self.rows),
        )
        token_index = np.fromiter(
            (row["token"] for row in self.rows), dtype=np.int64,
            count=len(self.rows),
        )
        out = {}
        for field_name in CACHE_FIELDS:
            for quantity in CACHE_SUMMARY_FIELDS:
                key = f"{field_name}_{quantity}"
                dtype = (np.uint8 if quantity == "cosine_valid" else np.float32)
                stacked = torch.stack(
                    [row[key] for row in self.rows]
                ).cpu().numpy()
                if quantity in ("before_norm", "after_norm"):
                    bad = ~np.isfinite(stacked)
                    if bad.any():
                        first = int(np.argmax(bad.reshape(len(self.rows), -1)
                                              .any(axis=1)))
                        raise MemoryPathError(
                            f"non-finite {key} at layer "
                            f"{int(layer_index[first])} token "
                            f"{int(token_index[first])}"
                        )
                packed = np.zeros((n_layers, n_tokens, nheads), dtype=dtype)
                packed[layer_index, token_index] = stacked.astype(
                    dtype, copy=False
                )
                out[key] = packed
        return out


# ---------------------------------------------------------------------------
# live prefill boundary capture
# ---------------------------------------------------------------------------


@dataclass
class PrefillBoundary:
    """Actual mixer inputs and outputs from one full-sequence forward."""

    inputs: dict = field(default_factory=dict)    # layer -> (T, D) float32 cpu
    outputs: dict = field(default_factory=dict)   # layer -> (T, D) float32 cpu
    input_fire_count: dict = field(default_factory=dict)
    output_fire_count: dict = field(default_factory=dict)
    source_dtype: str | None = None

    def release(self) -> None:
        self.inputs.clear()
        self.outputs.clear()


# ---------------------------------------------------------------------------
# per-step capture container
# ---------------------------------------------------------------------------


@dataclass
class PhaseCapture:
    """One phase (pre_update or post_update) of one captured step."""

    step: int
    phase: str
    boundary: PrefillBoundary = field(default_factory=PrefillBoundary)
    cache_arrays: dict = field(default_factory=dict)
    cache_retained_bytes: int = 0
    params: dict = field(default_factory=dict)     # layer -> fp32 cpu param dict
    specs: dict = field(default_factory=dict)      # layer -> InProjSpec
    arms: dict = field(default_factory=dict)       # layer -> bool is_mimo
    reference: dict = field(default_factory=dict)  # layer -> derived arrays
    parity: dict = field(default_factory=dict)     # layer -> metrics dict
    loss: float | None = None

    def release(self) -> None:
        self.boundary.release()
        self.params.clear()
        self.reference.clear()


@dataclass
class MemoryPathCapture:
    """Both phases of one captured optimizer update."""

    step: int
    input_ids: np.ndarray
    targets: np.ndarray
    phases: dict = field(default_factory=dict)
    stages_done: list = field(default_factory=list)

    def release(self) -> None:
        for phase in self.phases.values():
            phase.release()


# ---------------------------------------------------------------------------
# side-effect verification
# ---------------------------------------------------------------------------


class TrainerStateFingerprint:
    """Everything a replay must leave untouched.

    ``MemoryPathRecorder`` already guards RNG, training mode and parameter
    versions. The training integration owns the additional gradient and
    optimizer-state assertions, including ``None`` gradient identity.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.training = model.training
        # Identity plus version counter instead of a value clone.  ``_version``
        # bumps on any in-place write and object identity catches replacement,
        # so mutation is still detected exactly.  Cloning detected the same
        # thing at the cost of copying every parameter, every gradient and all
        # of AdamW's moments (~2.2GB here) twice per training step.
        self.param_versions = [p._version for p in model.parameters()]
        self.grads = [
            None if p.grad is None else (id(p.grad), p.grad._version)
            for p in model.parameters()
        ]
        self.state = {
            id(p): {
                k: ((id(v), v._version) if torch.is_tensor(v) else v)
                for k, v in optimizer.state[p].items()
            }
            for p in model.parameters() if p in optimizer.state
        }
        self.rng = tmpo.rng_snapshot()

    def assert_unchanged(self, where: str) -> None:
        if self.model.training != self.training:
            raise MemoryPathError(f"{where}: replay changed train/eval mode")
        params = list(self.model.parameters())
        for i, p in enumerate(params):
            if p._version != self.param_versions[i]:
                raise MemoryPathError(f"{where}: replay mutated a parameter value")
            before = self.grads[i]
            if before is None:
                if p.grad is not None:
                    raise MemoryPathError(
                        f"{where}: replay created a gradient where there was "
                        "none (None identity lost)"
                    )
            else:
                if p.grad is None:
                    raise MemoryPathError(f"{where}: replay destroyed a gradient")
                if (id(p.grad), p.grad._version) != before:
                    raise MemoryPathError(f"{where}: replay mutated a gradient")
        for p in params:
            if id(p) not in self.state:
                if p in self.optimizer.state and self.optimizer.state[p]:
                    raise MemoryPathError(
                        f"{where}: replay created optimizer state"
                    )
                continue
            live = self.optimizer.state[p]
            for key, value in self.state[id(p)].items():
                if key not in live:
                    raise MemoryPathError(
                        f"{where}: replay dropped optimizer state {key!r}"
                    )
                if isinstance(value, tuple):
                    observed = live[key]
                    if (not torch.is_tensor(observed)
                            or (id(observed), observed._version) != value):
                        raise MemoryPathError(
                            f"{where}: replay mutated optimizer state {key!r}"
                        )
                elif live[key] != value:
                    raise MemoryPathError(
                        f"{where}: replay mutated optimizer scalar {key!r}"
                    )
        current = tmpo.rng_snapshot()
        if not torch.equal(current["torch_cpu"], self.rng["torch_cpu"]):
            raise MemoryPathError(f"{where}: replay left CPU RNG state changed")
        if current["python"] != self.rng["python"]:
            raise MemoryPathError(f"{where}: replay left Python RNG state changed")
        if "torch_cuda" in self.rng and "torch_cuda" in current:
            for a, b in zip(current["torch_cuda"], self.rng["torch_cuda"]):
                if not torch.equal(a, b):
                    raise MemoryPathError(
                        f"{where}: replay left CUDA RNG state changed"
                    )


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------


class TrainingMemoryPathCamera:
    """Camera 4: live prefill + official cache summaries + reference pathway."""

    def __init__(self, model, optimizer, capture_steps=None,
                 cos_tol=fms.DEFAULT_COS_TOL, rel_tol=fms.DEFAULT_REL_TOL,
                 norm_max_tol=fms.DEFAULT_NORM_MAX_TOL,
                 rel_floor_frac=fms.DEFAULT_REL_FLOOR_FRAC):
        # Defaults bind at import; resolve here so the window can be retargeted.
        capture_steps = CAPTURE_WINDOW if capture_steps is None else capture_steps
        self.capture_steps = frozenset(int(s) for s in capture_steps)
        self.model = model
        self.optimizer = optimizer
        self.recorder = tmpo.MemoryPathRecorder(model)
        self.geometry = self.recorder.geometry
        self.mixers = self.recorder.mixers
        self._handles: list = []

        # Predeclared output parity thresholds, taken from the foundation.
        self.cos_tol = float(cos_tol)
        self.rel_tol = float(rel_tol)
        self.norm_max_tol = float(norm_max_tol)
        self.rel_floor_frac = float(rel_floor_frac)

        if self.recorder.state_claims_enabled:
            raise MemoryPathError(
                "recorder arrived with state claims enabled; Camera 4 v1 "
                f"refuses: {STATE_CLAIMS_REASON}"
            )

        # Realized per-layer in_proj geometry and SISO/MIMO arm, from tensors.
        ssm_cfg = None
        config = getattr(model, "config", None)
        if config is not None:
            raw = getattr(config, "ssm_cfg", None)
            if isinstance(raw, dict):
                ssm_cfg = raw
        self.specs = {}
        self.arms = {}
        for li, (name, mixer) in enumerate(self.mixers):
            spec = m3c.InProjSpec.from_mixer(mixer)
            spec.assert_against(int(mixer.in_proj.weight.shape[0]))
            self.specs[li] = spec
            # Tensor-derived arm is authoritative; config is a cross-check only.
            self.arms[li] = fms.resolve_arm(spec, ssm_cfg, where=name)

        # gamma_n for the out-projection contraction, from the REALIZED
        # contraction length. The decomposition reassociates one 1536-term
        # matmul into three, and gamma_n = n*u is the standard bound on the
        # rounding that reassociation can introduce. Derived from geometry, so
        # it cannot drift toward whatever a run happens to produce.
        contraction_lengths = {
            int(self.mixers[li][1].out_proj.weight.shape[1])
            for li in range(len(self.mixers))
        }
        if len(contraction_lengths) != 1:
            raise GeometryMismatchError(
                f"out_proj contraction length differs across layers: "
                f"{sorted(contraction_lengths)}"
            )
        self.closure_contraction_length = contraction_lengths.pop()
        self.closure_gamma = (
            self.closure_contraction_length * FLOAT32_UNIT_ROUNDOFF
        )
        global CLOSURE_GAMMA
        CLOSURE_GAMMA = self.closure_gamma

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def should_capture(self, step: int) -> bool:
        return int(step) in self.capture_steps

    def hooks_active(self) -> bool:
        return bool(self._handles)

    @property
    def claim_level(self) -> str:
        return self.recorder.claim_level

    def state_claims_enabled(self) -> bool:
        return bool(self.recorder.state_claims_enabled)

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------
    # live prefill boundary hooks
    # ------------------------------------------------------------------

    @contextmanager
    def prefill_hooks(self, boundary: PrefillBoundary, expected_tokens: int):
        """Hook every realized mixer's input and output for ONE forward.

        Each must fire exactly once. Clones are detached and compact; the hooks
        return ``None`` and never replace an output.
        """
        n_layers = len(self.mixers)
        d_model = int(self.geometry.d_model)

        def make_pre(li):
            def pre_hook(module, args):
                boundary.input_fire_count[li] = (
                    boundary.input_fire_count.get(li, 0) + 1
                )
                if boundary.input_fire_count[li] > 1:
                    raise MemoryPathError(
                        f"mixer {li} input hook fired "
                        f"{boundary.input_fire_count[li]} times; exactly one "
                        "prefill forward per phase is required"
                    )
                if not args or not torch.is_tensor(args[0]):
                    raise MemoryPathError(f"mixer {li} received no tensor input")
                u = args[0]
                self._check_boundary(u, li, "input", expected_tokens, d_model)
                boundary.inputs[li] = u.detach().float().cpu().clone()
                boundary.source_dtype = str(u.dtype)
                return None
            return pre_hook

        def make_post(li):
            def post_hook(module, args, output):
                boundary.output_fire_count[li] = (
                    boundary.output_fire_count.get(li, 0) + 1
                )
                if boundary.output_fire_count[li] > 1:
                    raise MemoryPathError(
                        f"mixer {li} output hook fired "
                        f"{boundary.output_fire_count[li]} times; exactly one "
                        "prefill forward per phase is required"
                    )
                if not torch.is_tensor(output):
                    raise MemoryPathError(
                        f"mixer {li} output is not a tensor "
                        f"({type(output).__name__})"
                    )
                self._check_boundary(output, li, "output", expected_tokens,
                                     d_model)
                boundary.outputs[li] = output.detach().float().cpu().clone()
                return None
            return post_hook

        try:
            for li, (_, mixer) in enumerate(self.mixers):
                self._handles.append(
                    mixer.register_forward_pre_hook(make_pre(li))
                )
                self._handles.append(
                    mixer.register_forward_hook(make_post(li))
                )
            yield boundary
        finally:
            self._remove_hooks()
            if len(boundary.inputs) not in (0, n_layers):
                raise MemoryPathError(
                    f"prefill captured {len(boundary.inputs)} of {n_layers} "
                    "mixer inputs"
                )

    @staticmethod
    def _check_boundary(tensor, li, which, expected_tokens, d_model):
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise GeometryMismatchError(
                f"mixer {li} {which} shape {tuple(tensor.shape)}, expected "
                "(1, T, D)"
            )
        if int(tensor.shape[1]) != expected_tokens:
            raise GeometryMismatchError(
                f"mixer {li} {which} has {int(tensor.shape[1])} tokens, "
                f"expected {expected_tokens}"
            )
        if d_model > 0 and int(tensor.shape[2]) != d_model:
            raise GeometryMismatchError(
                f"mixer {li} {which} width {int(tensor.shape[2])} != d_model "
                f"{d_model}"
            )
        if not torch.isfinite(tensor).all():
            raise NonFiniteCaptureError(
                f"non-finite value in mixer {li} live {which}"
            )

    def assert_boundary_complete(self, boundary: PrefillBoundary) -> None:
        n_layers = len(self.mixers)
        for which, store, counts in (
            ("input", boundary.inputs, boundary.input_fire_count),
            ("output", boundary.outputs, boundary.output_fire_count),
        ):
            if set(store) != set(range(n_layers)):
                raise MemoryPathError(
                    f"prefill {which}s captured for layers {sorted(store)}, "
                    f"expected 0..{n_layers - 1}"
                )
            bad = {li: c for li, c in counts.items() if c != 1}
            if bad:
                raise MemoryPathError(
                    f"prefill {which} hooks fired more than once: {bad}"
                )

    # ------------------------------------------------------------------
    # official cache replay
    # ------------------------------------------------------------------

    def record_cache_summary(self, input_ids, fingerprint: TrainerStateFingerprint,
                             where: str) -> tuple:
        """RNG-preserved token-by-token replay through the deployed step path.

        Returns (arrays, retained_bytes). The replay is asserted side-effect
        free on RNG, mode, parameters, gradients and optimizer state.
        """
        ids = (
            input_ids.detach().reshape(-1).cpu()
            if torch.is_tensor(input_ids) else torch.as_tensor(input_ids)
        )
        sink = TrainingCacheSummarySink()
        # The adapter exists ONLY for the duration of this replay. It is absent
        # during the live training forward, backward, clip, step, reference
        # derivation and off mode.
        with decode_shape_adapter(self.mixers):
            self.recorder.record_sequence(ids.tolist(), sink, metadata={
                "evidence_class": EVIDENCE_CACHE,
                "where": where,
                "decode_shape_adapter_applied": True,
            })
        if adapter_installed(self.mixers):
            raise DecodeShapeAdapterError(
                f"{where}: decode shape adapter survived the replay"
            )
        fingerprint.assert_unchanged(where)
        arrays = sink.to_arrays(
            n_layers=int(self.geometry.n_layers),
            n_tokens=int(ids.numel()),
            nheads=int(self.geometry.nheads),
        )
        return arrays, sink.retained_bytes()

    # ------------------------------------------------------------------
    # decode shape adapter preflight
    # ------------------------------------------------------------------

    def preflight_decode_shape_adapter(self, layer_index: int = 0,
                                       probe_tokens: int = 2) -> dict:
        """Prove the adapter is a pure rank change before any training runs.

        Both branches call the SAME original bound method with the same 2-D
        values and structurally identical zeroed caches, so the comparison is
        exact by construction. If exact equality fails, this aborts rather than
        widening a tolerance.

        Performs no optimizer update and uses fresh caches throughout.
        """
        label, mixer = self.mixers[layer_index]
        # Anchor the probe to in_proj.weight, the tensor u is actually
        # multiplied against, and the same anchor allocate_inference_cache uses.
        # next(mixer.parameters()) is NOT equivalent: in_proj is a submodule
        # while dt_bias / B_bias / C_bias / D are registered directly on the
        # mixer, so own-parameters come first, and several are float32 by
        # construction even in a bf16 model.
        weight = mixer.in_proj.weight
        device, dtype = weight.device, weight.dtype
        d_model = int(self.geometry.d_model)

        fingerprint = TrainerStateFingerprint(self.model, self.optimizer)
        rng = tmpo.rng_snapshot()
        try:
            with torch.inference_mode():
                generator = torch.Generator(device="cpu").manual_seed(20260808)
                u = torch.randn(
                    1, 1, d_model, generator=generator, dtype=torch.float32
                ).to(device=device, dtype=dtype)

                def fresh_cache():
                    cache = mixer.allocate_inference_cache(1, max(probe_tokens, 2))
                    for tensor in cache:
                        tensor.zero_()
                    return cache

                cache_direct = fresh_cache()
                cache_adapted = fresh_cache()
                for a, b in zip(cache_direct, cache_adapted):
                    if a.shape != b.shape or a.dtype != b.dtype:
                        raise DecodeShapeAdapterError(
                            "preflight cache sets are not structurally identical"
                        )

                original_step = mixer.step
                direct = original_step(u.squeeze(1), *cache_direct)
                with decode_shape_adapter([(label, mixer)]):
                    adapted = mixer.step(u, *cache_adapted)

                direct_out = direct[0]
                adapted_out = adapted[0]
                if adapted_out.dim() != 3 or int(adapted_out.shape[1]) != 1:
                    raise DecodeShapeAdapterError(
                        "preflight: adapted output did not regain a singleton "
                        f"axis, got {tuple(adapted_out.shape)}"
                    )
                if not torch.equal(direct_out, adapted_out.squeeze(1)):
                    raise DecodeShapeAdapterError(
                        "preflight: adapted output is not bit-identical to the "
                        "direct 2-D step output; aborting before training "
                        "rather than widening a tolerance"
                    )
                for name, a, b in zip(CACHE_FIELDS, direct[1:], adapted[1:]):
                    if not torch.equal(a, b):
                        raise DecodeShapeAdapterError(
                            f"preflight: returned {name} differs between the "
                            "direct and adapted branches"
                        )
                for name, a, b in zip(CACHE_FIELDS, cache_direct, cache_adapted):
                    if not torch.equal(a, b):
                        raise DecodeShapeAdapterError(
                            f"preflight: in-place mutated {name} differs "
                            "between the direct and adapted branches"
                        )

                # Short full-model replay under the adapter: coverage + cleanup.
                sink = TrainingCacheSummarySink()
                probe_ids = list(range(1, probe_tokens + 1))
                with decode_shape_adapter(self.mixers):
                    self.recorder.record_sequence(probe_ids, sink, metadata={
                        "evidence_class": EVIDENCE_CACHE,
                        "where": "decode shape adapter preflight",
                    })
                expected_rows = len(self.mixers) * probe_tokens
                if len(sink.rows) != expected_rows:
                    raise DecodeShapeAdapterError(
                        f"preflight replay covered {len(sink.rows)} rows, "
                        f"expected {expected_rows} (layers x tokens)"
                    )
                if adapter_installed(self.mixers):
                    raise DecodeShapeAdapterError(
                        "preflight: adapter survived the replay"
                    )
        finally:
            tmpo.rng_restore(rng)
        fingerprint.assert_unchanged("decode shape adapter preflight")

        return {
            "decode_shape_adapter_applied": True,
            "decode_shape_adapter_version": DECODE_SHAPE_ADAPTER_VERSION,
            "decode_shape_adapter_scope": DECODE_SHAPE_ADAPTER_SCOPE,
            "decode_shape_adapter_input": DECODE_SHAPE_ADAPTER_INPUT,
            "decode_shape_adapter_original_step_input": (
                DECODE_SHAPE_ADAPTER_ORIGINAL_STEP_INPUT
            ),
            "decode_shape_adapter_output_restored": (
                DECODE_SHAPE_ADAPTER_OUTPUT_RESTORED
            ),
            "decode_shape_adapter_recurrence_modified": False,
            "decode_shape_adapter_kernel_modified": False,
            "decode_shape_adapter_preflight_exact": True,
            "decode_shape_adapter_mixers": len(self.mixers),
            "decode_shape_adapter_preflight_layer": layer_index,
            "decode_shape_adapter_preflight_rows": len(sink.rows),
            "upstream_issue_record": UPSTREAM_ISSUE_RECORD,
        }

    # ------------------------------------------------------------------
    # phase-matched parameter basis
    # ------------------------------------------------------------------

    def snapshot_parameters(self, phase: PhaseCapture) -> None:
        """Freeze the exact CPU fp32 mixer parameter basis for this phase.

        Called for pre_update AFTER the backward but BEFORE clip/step, so the
        pre pathway is never derived with post-update weights.
        """
        for li in range(len(self.mixers)):
            extracted = fms.extract_layer_params(self.model, li)
            # extract_layer_params returns p.detach().float().cpu(). For a
            # parameter that is ALREADY fp32 on CPU, .float() and .cpu() are
            # both no-ops that return the same storage, so the "snapshot" would
            # alias the live parameter and the trainer's in-place
            # optimizer.step() would retroactively rewrite it. Cloning here is
            # what makes the snapshot genuinely phase-matched. (The foundation
            # file is provenance-bearing and must not be edited.)
            phase.params[li] = {
                name: tensor.clone() for name, tensor in extracted.items()
            }
            phase.specs[li] = self.specs[li]
            phase.arms[li] = self.arms[li]

    # ------------------------------------------------------------------
    # reference derivation
    # ------------------------------------------------------------------

    def derive_reference(self, phase: PhaseCapture) -> None:
        """Derive the reference pathway one layer at a time.

        The full returned dict is released after the declared sufficient basis
        is extracted, so all layers' transient state matrices are never retained
        simultaneously.
        """
        if not phase.params:
            raise MemoryPathError(
                f"step {phase.step} {phase.phase}: no phase-matched parameter "
                "snapshot; refusing to derive a pathway with mismatched weights"
            )
        for li in range(len(self.mixers)):
            if li not in phase.boundary.inputs:
                raise MemoryPathError(
                    f"step {phase.step} {phase.phase}: no live mixer input for "
                    f"layer {li}"
                )
            u = phase.boundary.inputs[li][0]          # (T, D) fp32 cpu
            params = phase.params[li]
            spec = phase.specs[li]
            a_floor = float(getattr(self.mixers[li][1], "A_floor", 1e-4))
            ref = refrec.reference_block_forward(
                u, params, spec, is_mimo=phase.arms[li], A_floor=a_floor,
                carried_phase=None, validate=True,
            )
            phase.reference[li] = self._sufficient_basis(phase, li, ref, params)
            phase.parity[li] = self._output_parity(phase, li, ref)
            del ref     # release transient state matrices before the next layer

    def _sufficient_basis(self, phase: PhaseCapture, li: int, ref: dict,
                          params: dict) -> dict:
        """Extract only the DESIGN fields.

        ``state_read``, ``prefill_accum``, ``gamma_write`` and ``deferred_write``
        are deliberately NOT persisted. The write matrices are exactly
        reconstructible from ``V_rank``, ``B_eff`` and their scalar
        coefficients, and the analyzer proves that reconstruction on fixtures.
        """
        basis = {}
        for name in REF_SCALAR_FIELDS:
            basis[name] = ref[name].detach().float().cpu().numpy()
        for name in ("B_eff", "C_eff", "V_rank"):
            basis[name] = ref[name].detach().float().cpu().numpy()
        for pathway in REF_PATHWAYS:
            basis[f"y_{pathway}_per_rank"] = (
                ref[f"y_{pathway}_per_rank"].detach().float().cpu().numpy()
            )
        basis["gate_factor"] = ref["gate_factor"].detach().float().cpu().numpy()
        collapse = ref["collapse_weight"].detach().float().cpu()
        basis["collapse_weight"] = collapse.numpy()

        # Write factors: |scale| * ||V|| * ||B_eff||, per (T, H, R). This is the
        # exact Frobenius norm of the rank-one outer product scale*V (x) B, so
        # the dense write matrices never need to exist.
        v_norm = torch.linalg.vector_norm(ref["V_rank"].detach().float(), dim=-1)
        b_norm = torch.linalg.vector_norm(ref["B_eff"].detach().float(), dim=-1)
        gamma = ref["gamma"].detach().float().abs().unsqueeze(-1)
        shifted = ref["shifted_gamma"].detach().float().abs().unsqueeze(-1)
        basis["immediate_write_norm"] = (gamma * v_norm * b_norm).cpu().numpy()
        basis["deferred_write_norm"] = (shifted * v_norm * b_norm).cpu().numpy()

        T = int(ref["out"].shape[0])
        # Resolution bookkeeping: the immediate half resolves at the source
        # token, the deferred half at the next token (undefined at the last).
        basis["write_source_token"] = np.arange(T, dtype=np.int64)
        basis["immediate_resolution_token"] = np.arange(T, dtype=np.int64)
        deferred = np.arange(1, T + 1, dtype=np.int64)
        deferred[-1] = -1          # explicit "no successor" marker
        basis["deferred_resolution_token"] = deferred

        # Model-space contributions: apply the SAME gate, collapse and
        # projection the reference applied, per pathway. Their sum must equal
        # the reference output -- an algebraic decomposition, not intervention.
        out_w = params["out_proj.weight"]
        gate = ref["gate_factor"].detach().float()
        model_space = {}
        abs_products = []
        for pathway in REF_PATHWAYS:
            y_p = ref[f"y_{pathway}_per_rank"].detach().float()
            collapsed = (y_p * gate * collapse).sum(dim=2)      # (T, H, P)
            flat = collapsed.reshape(T, -1)
            contribution = flat @ out_w.T                       # (T, d_model)
            model_space[pathway] = contribution
            basis[f"model_space_{pathway}"] = contribution.numpy()
            # |A| @ |B| for this pathway: the conditioning scale of the
            # contraction, per the matmul backward-error bound.
            abs_products.append(flat.abs() @ out_w.abs().T)

        reference_out = ref["out"].detach().float()
        basis["mixer_output_reference"] = reference_out.numpy()

        total = model_space["earlier"] + model_space["diagonal"] + \
            model_space["feedthrough"]
        within, rel_dev, abs_gap = closure_deviation(
            total, reference_out, abs_products
        )
        if not within:
            raise MemoryPathError(
                f"step {phase.step} {phase.phase} layer {li}: model-space "
                f"earlier+diagonal+feedthrough does not close on the reference "
                f"mixer output (max relative deviation {rel_dev}, max abs gap "
                f"{abs_gap}, measured against the summed term magnitudes)"
            )
        # Recorded, not just gated: the closure quality is visible per layer.
        basis["closure_max_rel_dev"] = np.float32(rel_dev).reshape(())
        basis["closure_max_abs_gap"] = np.float32(abs_gap).reshape(())
        for name, arr in basis.items():
            if np.issubdtype(np.asarray(arr).dtype, np.floating) and \
                    not np.all(np.isfinite(arr)):
                raise NonFiniteCaptureError(
                    f"step {phase.step} {phase.phase} layer {li}: non-finite "
                    f"value in reference basis {name}"
                )
        return basis

    def _output_parity(self, phase: PhaseCapture, li: int, ref: dict) -> dict:
        """Full-sequence live vs reference mixer-output parity for one layer.

        Uses the foundation's own metric grain and predeclared thresholds. The
        reference tensor is the candidate; the live deployed output is the
        comparison, matching the foundation's convention.
        """
        live = phase.boundary.outputs[li][0]              # (T, D) fp32 cpu
        candidate = ref["out"].detach().float().cpu()
        if candidate.shape != live.shape:
            raise GeometryMismatchError(
                f"step {phase.step} {phase.phase} layer {li}: reference output "
                f"{tuple(candidate.shape)} != live output {tuple(live.shape)}"
            )
        metrics = fms.parity_metrics(candidate, live,
                                     rel_floor_frac=self.rel_floor_frac)
        verdict = fms.verdict_for(metrics, self.cos_tol, self.rel_tol,
                                  self.norm_max_tol)
        return {
            "layer": li,
            "verdict": verdict,
            "cosine": float(metrics["cosine"]),
            "max_abs_err": float(metrics["max_abs_err"]),
            "norm_max_err": float(metrics["norm_max_err"]),
            "vendor_rel_p95": float(metrics["vendor_rel_p95"]),
        }

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def capture(self, step: int, input_ids, targets):
        if not self.should_capture(step):
            raise MemoryPathError(
                f"capture requested for step {step}, outside window "
                f"{sorted(self.capture_steps)}"
            )
        ids = np.asarray(
            input_ids.detach().cpu().numpy() if torch.is_tensor(input_ids)
            else input_ids
        ).reshape(-1).astype(np.int64)
        tgt = np.asarray(
            targets.detach().cpu().numpy() if torch.is_tensor(targets)
            else targets
        ).reshape(-1).astype(np.int64)
        if ids.size != tgt.size:
            raise MemoryPathError(
                f"step {step}: input_ids has {ids.size} positions but targets "
                f"has {tgt.size}; the control's causal contract is "
                "input_ids = ids[:-1], targets = ids[1:]"
            )
        cap = MemoryPathCapture(step=int(step), input_ids=ids, targets=tgt)
        try:
            yield cap
        finally:
            self._remove_hooks()
            cap.release()

    def begin_phase(self, cap: MemoryPathCapture, phase: str) -> PhaseCapture:
        if phase not in PHASES:
            raise MemoryPathError(f"unknown phase {phase!r}; expected {PHASES}")
        if phase in cap.phases:
            raise MemoryPathError(
                f"step {cap.step}: phase {phase!r} already started"
            )
        expected = PHASES[len(cap.phases)]
        if phase != expected:
            raise MemoryPathError(
                f"step {cap.step}: phase {phase!r} out of order; expected "
                f"{expected!r}"
            )
        pc = PhaseCapture(step=cap.step, phase=phase)
        cap.phases[phase] = pc
        return pc

    def assert_complete(self, cap: MemoryPathCapture) -> None:
        if list(cap.phases) != list(PHASES):
            raise MemoryPathError(
                f"step {cap.step}: phases {list(cap.phases)} != {list(PHASES)}"
            )
        n_layers = len(self.mixers)
        for phase_name, phase in cap.phases.items():
            self.assert_boundary_complete(phase.boundary)
            if set(phase.reference) != set(range(n_layers)):
                raise MemoryPathError(
                    f"step {cap.step} {phase_name}: reference derived for "
                    f"layers {sorted(phase.reference)}, expected all {n_layers}"
                )
            if set(phase.parity) != set(range(n_layers)):
                raise MemoryPathError(
                    f"step {cap.step} {phase_name}: parity missing layers"
                )
            if not phase.cache_arrays:
                raise MemoryPathError(
                    f"step {cap.step} {phase_name}: no official cache summary"
                )
            if phase.loss is None or not np.isfinite(phase.loss):
                raise MemoryPathError(
                    f"step {cap.step} {phase_name}: phase loss missing or "
                    "non-finite"
                )
        if self.hooks_active():
            raise MemoryPathError(f"step {cap.step}: hook handles survived")

    # ------------------------------------------------------------------
    # derivation into storable arrays
    # ------------------------------------------------------------------

    def derive(self, cap: MemoryPathCapture, phase_name: str, token_ids, *,
               dataset_row: int, optimizer_step: int) -> tto.DerivedCapture:
        phase = cap.phases[phase_name]
        arrays: dict = {}
        records: list = []
        raw_bytes = 0

        def _record(key, name, arr, evidence, source_dtype, storage_dtype,
                    layer=None, may_be_nan=False):
            nonlocal raw_bytes
            arr = np.asarray(arr)
            if arr.ndim > 0:
                arr = np.ascontiguousarray(arr)
            if np.issubdtype(arr.dtype, np.floating):
                # Cosine fields carry masked NaN where the quantity is genuinely
                # undefined (a zero-norm cache at token 0). Their paired
                # *_cosine_valid mask is the authority; zero-filling them would
                # assert orthogonality.
                bad = (~np.isfinite(arr) if not may_be_nan
                       else np.isinf(arr))
                if np.any(bad):
                    raise NonFiniteCaptureError(
                        f"step {cap.step} {phase_name}: non-finite value in {key}"
                    )
            arrays[key] = arr
            raw_bytes += arr.nbytes
            records.append({
                "key": key,
                "layer": layer,
                "name": name,
                "evidence_class": evidence,
                "shape": list(arr.shape),
                "source_dtype": source_dtype,
                "storage_dtype": storage_dtype,
                "finite_count": int(arr.size),
                "nonfinite_count": 0,
                "min": None, "max": None, "mean": None, "std": None,
                "raw_bytes": int(arr.nbytes),
            })

        n_layers = len(self.mixers)
        for li in range(n_layers):
            tag = f"layer_{li:02d}"
            # -- live prefill boundaries
            _record(f"{tag}/live/mixer_input", "mixer_input",
                    phase.boundary.inputs[li][0].numpy(), EVIDENCE_LIVE,
                    phase.boundary.source_dtype or "unknown", "float32", li)
            _record(f"{tag}/live/mixer_output_live", "mixer_output_live",
                    phase.boundary.outputs[li][0].numpy(), EVIDENCE_LIVE,
                    phase.boundary.source_dtype or "unknown", "float32", li)
            # -- reference-derived sufficient basis
            for name, arr in phase.reference[li].items():
                dtype = "int64" if np.asarray(arr).dtype == np.int64 else "float32"
                _record(f"{tag}/ref/{name}", name, arr, EVIDENCE_REFERENCE,
                        "float32", dtype, li)
            # -- live/reference parity, per layer
            metrics = phase.parity[li]
            for name in ("cosine", "max_abs_err", "norm_max_err",
                         "vendor_rel_p95"):
                _record(f"{tag}/parity/{name}", f"parity_{name}",
                        np.float32(metrics[name]).reshape(()), EVIDENCE_LIVE,
                        "float32", "float32", li)

        # -- official cache summaries, shaped [L, T, H]
        for name, arr in phase.cache_arrays.items():
            _record(f"cache/{name}", name, arr, EVIDENCE_CACHE,
                    "official_cache", str(np.asarray(arr).dtype),
                    may_be_nan=name.endswith("_cosine"))

        ids = [int(i) for i in token_ids]
        supervised = len(ids) - 1
        if supervised != int(cap.input_ids.size):
            raise MemoryPathError(
                f"step {cap.step}: supervised_token_count {supervised} != "
                f"{int(cap.input_ids.size)} captured positions"
            )
        for key, arr, dtype in (
            ("token_ids", np.asarray(ids, dtype="<i8"), "int64"),
            ("input_ids", cap.input_ids, "int64"),
            ("targets", cap.targets, "int64"),
            ("supervised_token_count", np.asarray(supervised, dtype="<i8"),
             "int64"),
            ("dataset_row", np.asarray(int(dataset_row), dtype="<i8"), "int64"),
            ("optimizer_step", np.asarray(int(optimizer_step), dtype="<i8"),
             "int64"),
            ("state_claims_enabled", np.asarray(0, dtype="<i8"), "int64"),
        ):
            _record(key, key, arr, EVIDENCE_LIVE, dtype, dtype)
        _record("phase_loss", "phase_loss",
                np.float32(phase.loss).reshape(()), EVIDENCE_LIVE,
                "float32", "float32")
        _record("cache_retained_bytes", "cache_retained_bytes",
                np.asarray(int(phase.cache_retained_bytes), dtype="<i8"),
                EVIDENCE_CACHE, "int64", "int64")

        return tto.DerivedCapture(
            step=cap.step,
            phase=phase_name,
            arrays=arrays,
            tensor_records=records,
            token_ids=ids,
            token_sha256=token_ids_sha256(ids),
            sequence_length=supervised,
            raw_bytes=raw_bytes,
        )


# ---------------------------------------------------------------------------
# compact W&B scalar summaries
# ---------------------------------------------------------------------------


def camera_scalars(derived: tto.DerivedCapture, phase: str) -> dict:
    a = derived.arrays
    cosines = [float(v) for k, v in a.items() if k.endswith("/parity/cosine")]
    rel = [float(v) for k, v in a.items() if k.endswith("/parity/vendor_rel_p95")]
    norm_max = [float(v) for k, v in a.items() if k.endswith("/parity/norm_max_err")]
    prefix = f"observer/memory_path/{phase}"
    out = {
        f"{prefix}/loss": float(a["phase_loss"]),
        f"{prefix}/parity_cosine_min": min(cosines) if cosines else 0.0,
        f"{prefix}/parity_vendor_rel_p95_max": max(rel) if rel else 0.0,
        f"{prefix}/parity_norm_max_err_max": max(norm_max) if norm_max else 0.0,
        f"{prefix}/cache_retained_bytes": int(a["cache_retained_bytes"]),
        "observer/memory_path/state_claims_enabled": 0,
    }
    for family in CACHE_FIELDS:
        key = f"cache/{family}_delta_norm"
        if key in a:
            out[f"{prefix}/{family}_delta_norm_mean"] = float(np.mean(a[key]))
    for pathway in REF_PATHWAYS:
        norms = [
            float(np.sqrt(np.sum(np.square(v))))
            for k, v in a.items() if k.endswith(f"/ref/model_space_{pathway}")
        ]
        out[f"{prefix}/model_space_{pathway}_l2"] = (
            float(np.sqrt(np.sum(np.square(norms)))) if norms else 0.0
        )
    return out


# ---------------------------------------------------------------------------
# artifact writing
# ---------------------------------------------------------------------------


def expected_filenames() -> list:
    return [
        f"step_{step:06d}_{phase}.npz"
        for step in range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1)
        for phase in PHASES
    ]


def finalize_memory_path_dir(out_dir, file_records: list, run_meta: dict,
                             expected_files: int = 12) -> dict:
    out_dir = Path(out_dir)
    if len(file_records) != expected_files:
        raise MemoryPathError(
            f"expected {expected_files} capture files, have {len(file_records)}"
        )
    file_records = sorted(file_records, key=lambda r: r["filename"])
    if sorted(r["filename"] for r in file_records) != sorted(expected_filenames()):
        raise MemoryPathError(
            f"capture filenames {[r['filename'] for r in file_records]} != "
            f"expected {sorted(expected_filenames())}"
        )

    by_quantity: dict = {}
    by_evidence: dict = {}
    for r in file_records:
        for t in r["tensors"]:
            agg = by_quantity.setdefault(
                t["name"],
                {"name": t["name"], "evidence_class": t.get("evidence_class"),
                 "count": 0, "raw_bytes": 0, "npy_bytes": 0,
                 "compressed_bytes": 0},
            )
            agg["count"] += 1
            agg["raw_bytes"] += t["raw_bytes"]
            agg["npy_bytes"] += t.get("npy_bytes", 0)
            agg["compressed_bytes"] += t.get("compressed_bytes", 0)
            ev = by_evidence.setdefault(
                t.get("evidence_class"),
                {"evidence_class": t.get("evidence_class"), "count": 0,
                 "raw_bytes": 0, "compressed_bytes": 0},
            )
            ev["count"] += 1
            ev["raw_bytes"] += t["raw_bytes"]
            ev["compressed_bytes"] += t.get("compressed_bytes", 0)
    ranked = sorted(by_quantity.values(),
                    key=lambda a: a["compressed_bytes"], reverse=True)

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
                        "evidence_class": t.get("evidence_class"),
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
        "by_quantity": ranked,
        "by_evidence_class": sorted(
            by_evidence.values(), key=lambda a: a["compressed_bytes"],
            reverse=True,
        ),
        "total_raw_bytes": sum(r["raw_bytes"] for r in file_records),
        "total_npy_payload_bytes": sum(r["npy_payload_bytes"] for r in file_records),
        "total_compressed_payload_bytes": sum(
            r["compressed_payload_bytes"] for r in file_records
        ),
        "total_archive_bytes": sum(r["archive_bytes"] for r in file_records),
        "total_compressed_bytes": sum(r["compressed_bytes"] for r in file_records),
    }
    storage_path = out_dir / "storage_report.json"
    storage_path.write_text(
        json.dumps(storage_report, sort_keys=True, indent=2) + "\n"
    )

    hashes = {r["filename"]: sha256_file(out_dir / r["filename"])
              for r in file_records}
    hashes["storage_report.json"] = sha256_file(storage_path)

    manifest = {
        "memory_path_camera_schema_version": MEMORY_PATH_CAMERA_SCHEMA_VERSION,
        "camera_number": CAMERA_NUMBER,
        "camera_family": CAMERA_FAMILY,
        "phases": list(PHASES),
        "evidence_classes": list(EVIDENCE_CLASSES),
        "evidence_class_language": {
            EVIDENCE_LIVE: (
                "actual mixer inputs/outputs from the real training forward and "
                "the identical-example post-update replay"
            ),
            EVIDENCE_CACHE: (
                OFFICIAL_CACHE_EVIDENCE_LABEL
                + "; these are cache magnitude/direction summaries ONLY. No "
                "prose may say a token is stored or retained for a duration on "
                "their basis"
            ),
            EVIDENCE_REFERENCE: (
                "source-derived sufficient basis and write/read/output "
                "decomposition from reference_block_forward; algebraic shares, "
                "not intervention effects"
            ),
        },
        "state_claims_enabled": False,
        "state_claims_reason": STATE_CLAIMS_REASON,
        "parity_source_run": PARITY_RUN_ID,
        "parity_verdict": PARITY_VERDICT,
        "prohibited_claims": [
            "h_t", "causal-state trajectory", "retention duration",
            "kernel-state equivalence",
        ],
        "fields_omitted_because_reconstructible": [
            "state_read", "prefill_accum", "gamma_write", "deferred_write",
        ],
        "write_factor_reconstruction": (
            "immediate_write_norm = |gamma| * ||V_rank|| * ||B_eff|| and "
            "deferred_write_norm = |shifted_gamma| * ||V_rank|| * ||B_eff||, "
            "per (token, head, rank); these are the exact Frobenius norms of "
            "the rank-one outer products, so the dense write matrices are "
            "never materialized"
        ),
        "output_decomposition": (
            "model_space_earlier + model_space_diagonal + "
            "model_space_feedthrough == mixer_output_reference, after the "
            "shared gate, rank collapse and the phase-matched out projection; "
            "an algebraic decomposition of the reference output, not a causal "
            "intervention"
        ),
        "output_parity_thresholds": run_meta.get("output_parity_thresholds"),
        # Decode shape adapter: rank plumbing only, scoped to cache replay.
        "decode_shape_adapter_applied": True,
        "decode_shape_adapter_version": DECODE_SHAPE_ADAPTER_VERSION,
        "decode_shape_adapter_scope": DECODE_SHAPE_ADAPTER_SCOPE,
        "decode_shape_adapter_input": DECODE_SHAPE_ADAPTER_INPUT,
        "decode_shape_adapter_original_step_input": (
            DECODE_SHAPE_ADAPTER_ORIGINAL_STEP_INPUT
        ),
        "decode_shape_adapter_output_restored": DECODE_SHAPE_ADAPTER_OUTPUT_RESTORED,
        "decode_shape_adapter_recurrence_modified": False,
        "decode_shape_adapter_kernel_modified": False,
        "official_cache_evidence_label": OFFICIAL_CACHE_EVIDENCE_LABEL,
        "upstream_issue_record": UPSTREAM_ISSUE_RECORD,
        "upstream_issue_url": UPSTREAM_ISSUE_URL,
        "train_step_time_comparable_to_control": False,
        "train_step_time_note": (
            "the official pre-update cache replay and the pre parameter "
            "transfer occur before the step-time freeze, so train/step_time "
            "includes their overhead and is NOT comparable to the earlier "
            "cameras; observer durations are logged separately and no "
            "corrected timing estimate is invented"
        ),
        "capture_window": sorted({r["step"] for r in file_records}),
        "finite_assertions_passed": True,
        "files": [
            {**{k: v for k, v in r.items()}, "sha256": hashes[r["filename"]]}
            for r in file_records
        ],
        "storage_report_sha256": hashes["storage_report.json"],
        "total_raw_bytes": storage_report["total_raw_bytes"],
        "total_compressed_bytes": storage_report["total_compressed_bytes"],
        "total_compressed_payload_bytes": storage_report[
            "total_compressed_payload_bytes"
        ],
        **run_meta,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    hashes["manifest.json"] = sha256_file(manifest_path)
    lines = [f"{hashes[name]}  {name}" for name in sorted(hashes)]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    return manifest
