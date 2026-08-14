"""Training optimizer-delivery observer for Mamba-3 (Camera 3, steps 20-25).

Modal-free by design so every mechanism is CPU-testable. The Modal trainer
(modal_optimizer_delivery_observer.py) imports this module; nothing here
imports modal, wandb, or CUDA-only code.

Camera 3 films the middle of the ask -> deliver -> respond loop. Camera 2
records what training ASKED for (pre-clip activation gradients at the control
surface); Camera 1 records the RESPONSE (post-update recurrence surfaces).
Camera 3 records the DELIVERY: the parameter gradients left by the one
ordinary ``loss.backward()``, the clip event, the AdamW update, and the
realized parameter delta.

Read-only by construction
-------------------------
Camera 3 registers NO module hooks and NO tensor gradient hooks. It takes
detached clones of ``.grad``, optimizer state, and parameters. It never calls
``clip_grad_norm_``, ``optimizer.step()``, or ``backward()`` -- the trainer
owns all three. Camera 2 must prove its tensor hooks return ``None``; Camera 3
has no hooks to prove neutral, so ``hooks_active()`` is False by construction
and the passivity argument is structural rather than behavioural.

Three staged snapshots, never conflated
---------------------------------------
``pre_clip_parameter_gradient`` -> ``post_clip_parameter_gradient`` ->
``realized_parameter_delta``. The capture context is entered AFTER
``loss.backward()`` and brackets the trainer's clip and step calls::

    with observer.capture(step) as cap:
        observer.record_pre_clip(cap)          # after backward, before clip
        grad_norm = clip_grad_norm_(...)       # trainer, not observer
        observer.record_clip(cap, grad_norm)   # after clip, before step
        optimizer.step()                       # trainer, not observer
        step_time = ...                        # control's timing boundary
        observer.record_post_step(cap)         # after step
    observer.assert_complete(cap)

Measurement decisions recorded here, so the analyzer can never over-claim
-------------------------------------------------------------------------
* ``delta_precision = float32_subtraction_of_native_params``. Parameters are
  bfloat16 in the frozen control; pre- and post-step values are upcast to
  float32 and subtracted. Deltas below the bf16 step resolution record as
  exactly zero and are reported as ``delta_fraction_zero`` -- never imputed,
  never denoised, never averaged through (MaskedMoments doctrine). The fp32
  reference is an idealization; the delta is what the bf16 optimizer actually
  delivered to bf16 parameters, measured in fp32.
* ``clip_coefficient = expected_and_empirical``. Three independent views:
  the observer's own pre-clip global norm, the norm returned by the trainer's
  real ``clip_grad_norm_``, and the empirical post/pre global ratio. The
  expected coefficient is ``min(1, max_norm / (returned_norm + 1e-6))``.
  Material disagreement raises. Coordinate-wise ratios are never used as the
  estimator: zero gradients and bf16 rounding make them unstable.
* ``adamw_decomposition = fp32_semantic_reconstruction``. Idealized fp32
  decoupled decay, idealized fp32 adaptive term, their sum, the realized
  native delta, and a REALIZATION RESIDUAL. The residual contains native
  moment/denominator/backend arithmetic as well as parameter rounding; it is
  never labeled pure bf16 quantization. No scalar is ever called an
  "effective learning rate": AdamW's elementwise normalization has none.
* Scalar delivery stats for EVERY optimizer parameter, which requires a
  temporary native-dtype pre-step clone of every optimizer parameter. It is
  held only until the post-step scalars are computed, then released.
* Full vectors ONLY for the control-relevant slices: the ``in_proj.weight``
  ROWS that produce dd_dt / dd_A / trap / angles (the same
  ``InProjSpec.bounds`` indices Camera 2 applies to the in_proj OUTPUT, here
  applied to the in_proj WEIGHT ROWS), plus ``dt_bias``, ``D`` and
  ``norm.weight`` when present. Mamba-3 has no trainable ``A_log``; ``dd_A``
  is emitted by ``in_proj`` and ``A_floor`` is a non-parameter scalar.
* The roster is optimizer-centric, not name-centric: stable identity,
  canonical name, ALL tied aliases, group index/offset, shape, source dtype,
  ``grad_present``, ``state_present``, detail scope, and any
  ``_no_weight_decay`` hint. The optimizer group is authoritative; a
  model-side hint is metadata only. A parameter with ``grad is None`` is
  skipped by AdamW, including its decoupled decay.

Storage layout decisions
------------------------
Per-parameter scalars are stored as 1-D arrays indexed by roster order
(float64 for numeric stats, int64 for counts/flags), with the roster itself
carried in the manifest. Each NPZ additionally stores ``parameter_names`` as a
native unicode array so a bundle is self-describing and the analyzer can
cross-check NPZ ordering against the manifest roster -- a real guard against
roster/array misalignment. Detail vectors are stored per layer and region.

Derived quantities that are genuinely undefined are NOT stored with a stand-in
value. Delivery alignment is stored as ``delivery_inner_product`` alongside
the two norms; the analyzer forms a cosine only where both norms are nonzero.
Storing 0.0 for an undefined cosine would mean "orthogonal", which is
imputation.

Infrastructure reuse: mixer discovery, per-layer ``InProjSpec`` geometry,
cross-layer geometry assertion, ``A_floor`` presence, deterministic NPZ
writing, tensor records, token hashing, and the non-finite policy all come
from ``training_time_observer`` (Camera 1, untouched), exactly as Camera 2
reuses them. No second in-projection geometry implementation exists here.
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

# Canonical module identities (mamba3_core registration) are established by
# this import; Camera 3 reuses Camera 1's discovery, spec, writer, and errors.
import training_time_observer as tto  # noqa: E402

ObserverError = tto.ObserverError
NonFiniteCaptureError = tto.NonFiniteCaptureError
GeometryMismatchError = tto.GeometryMismatchError

write_step_npz = tto.write_step_npz
sha256_file = tto.sha256_file
token_ids_sha256 = tto.token_ids_sha256

OPTIMIZER_DELIVERY_SCHEMA_VERSION = 1
CAMERA_NUMBER = 3
CAMERA_FAMILY = "optimizer_delivery"
CAPTURE_START_STEP = 20
CAPTURE_END_STEP = 25
CAPTURE_WINDOW = frozenset(range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1))
PHASE = "optimizer_delivery"  # -> step_XXX_optimizer_delivery.npz

OBSERVATION_STAGE = "optimizer_delivery"
GRADIENT_STAGE_SPLIT = (
    "pre_clip_parameter_gradient",
    "post_clip_parameter_gradient",
    "realized_parameter_delta",
)
DELTA_PRECISION = "float32_subtraction_of_native_params"
CLIP_COEFFICIENT_SEMANTICS = "expected_and_empirical"
ADAMW_DECOMPOSITION_SEMANTICS = "fp32_semantic_reconstruction"
SCALAR_STAT_PRECISION = "float32_compute_float64_storage"

# The four canonical in_proj control regions. Camera 2 slices the in_proj
# OUTPUT with these bounds; Camera 3 slices the in_proj WEIGHT ROWS -- the
# rows that PRODUCE those output features. Same bounds, different axis.
CONTROL_REGIONS = ("dd_dt", "dd_A", "trap", "angles")

# Adjacent mixer parameters stored as whole tensors when present. dt_bias is
# in the Camera-1/2 recurrence-control path; D and norm.weight are adjacent
# context and must NOT be described as coordinate-wise Camera-2 matches.
FULL_TENSOR_PARAMS = ("dt_bias", "D", "norm_weight")

DETAIL_SCOPES = ("scalar_only", "row_slice", "full_tensor")
DETAIL_SCOPE_CODES = {name: i for i, name in enumerate(DETAIL_SCOPES)}

# Sufficient basis retained per detailed tensor. The analyzer derives the four
# delivery components offline from these.
DETAIL_FIELDS = (
    "param_pre",
    "grad_pre_clip",
    "grad_post_clip",
    "exp_avg_pre",
    "exp_avg_sq_pre",
    "exp_avg_post",
    "exp_avg_sq_post",
    "delta",
)

# Per-parameter scalar stats, grouped by the stage that produces them. Field
# order is fixed so one batched device->host copy per stage suffices.
PRE_CLIP_SCALAR_FIELDS = (
    "pre_clip_grad_l2",
    "pre_clip_grad_max_abs",
    "param_pre_l2",
    "param_pre_max_abs",
    "exp_avg_pre_l2",
    "exp_avg_sq_pre_l2",
)
CLIP_SCALAR_FIELDS = (
    "post_clip_grad_l2",
    "post_clip_grad_max_abs",
)
POST_STEP_SCALAR_FIELDS = (
    "param_post_l2",
    "param_post_max_abs",
    "exp_avg_post_l2",
    "exp_avg_sq_post_l2",
    "delta_l2",
    "delta_max_abs",
    "delta_fraction_zero",
    "ideal_decay_l2",
    "ideal_adaptive_l2",
    "ideal_total_l2",
    "realization_residual_l2",
    "realization_residual_max_abs",
    "delivery_inner_product",
)
PARAM_SCALAR_FIELDS = (
    PRE_CLIP_SCALAR_FIELDS + CLIP_SCALAR_FIELDS + POST_STEP_SCALAR_FIELDS
)

# Python-side per-parameter values; no device round trip needed.
PARAM_GROUP_FLOAT_FIELDS = (
    "group_lr",
    "group_weight_decay",
    "group_beta1",
    "group_beta2",
    "group_eps",
    "bias_correction1",
    "bias_correction2",
)
# state_step_pre / state_step_post use -1 when no optimizer state exists.
# The paired *_present flag is the authority; -1 is an explicit absence
# marker for an integer field, not an imputed measurement.
PARAM_META_INT_FIELDS = (
    "numel",
    "group_index",
    "group_offset",
    "grad_present",
    "state_present_pre",
    "state_present_post",
    "state_step_pre",
    "state_step_post",
    "detail_scope_code",
    "no_weight_decay_hint",
    "amsgrad",
    "maximize",
)

CLIP_GLOBAL_FLOAT_FIELDS = (
    "clip_max_norm",
    "observer_pre_clip_total_norm",
    "returned_total_norm",
    "post_clip_total_norm",
    "clip_coefficient_expected",
    "clip_coefficient_empirical",
    "observer_pre_clip_total_norm_optimizer_scope",
    "post_clip_total_norm_optimizer_scope",
)
CLIP_GLOBAL_INT_FIELDS = (
    "clip_fired",
    "clip_scope_matches_optimizer_scope",
)

GLOBAL_NAMES = (
    "token_ids",
    "training_loss",
    "supervised_token_count",
    "dataset_row",
    "optimizer_step",
    "parameter_count",
    "parameter_names",
)

STAGES = ("pre_clip", "clip", "post_step")

# torch's clip_grad_norm_ reduces in the gradient's own dtype, so the norm it
# returns carries that dtype's rounding; the observer reduces in float32 for
# resolution. The cross-check tolerance is therefore derived from the COARSEST
# gradient dtype in the clip set, not tuned until a test passes.
#   bfloat16: eps = 2^-8  ~= 3.9e-3, accumulated over a norm-of-norms tree
#   float16 : eps = 2^-11 ~= 4.9e-4
#   float32 : eps = 2^-24 ~= 6.0e-8
_DTYPE_NORM_RTOL = {
    torch.bfloat16: 5e-2,
    torch.float16: 1e-2,
    torch.float32: 1e-4,
    torch.float64: 1e-9,
}
_DEFAULT_NORM_RTOL = 5e-2
NORM_CHECK_ATOL = 1e-6

# Clipping rule as implemented by torch.nn.utils.clip_grad_norm_.
CLIP_EPS = 1e-6


def clip_coefficient_expected(max_norm: float, returned_norm: float) -> float:
    """Exact expected coefficient: ``min(1, max_norm/(returned + 1e-6))``.

    Derived from the clipping definition, never fitted to observed output.
    """
    return float(min(1.0, float(max_norm) / (float(returned_norm) + CLIP_EPS)))


def norm_rtol_for_dtype(dtype) -> float:
    return _DTYPE_NORM_RTOL.get(dtype, _DEFAULT_NORM_RTOL)


def _close(a: float, b: float, rtol: float, atol: float = NORM_CHECK_ATOL) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


# ---------------------------------------------------------------------------
# roster
# ---------------------------------------------------------------------------


@dataclass
class ParamRecord:
    """One optimizer parameter, identified by optimizer membership.

    ``aliases`` holds every model-side name bound to the same tensor, so tied
    weights collapse to ONE optimizer row instead of duplicating.
    """

    index: int
    parameter_id: str
    canonical_name: str
    aliases: tuple
    group_index: int
    group_offset: int
    shape: tuple
    source_dtype: str
    numel: int
    no_weight_decay_hint: bool
    detail_scope: str
    detail_layer: int | None = None
    detail_kind: str | None = None
    detail_storage_name: str | None = None
    param: torch.nn.Parameter | None = None

    def as_manifest_entry(self) -> dict:
        return {
            "index": self.index,
            "parameter_id": self.parameter_id,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "group_index": self.group_index,
            "group_offset": self.group_offset,
            "shape": list(self.shape),
            "source_dtype": self.source_dtype,
            "numel": self.numel,
            "no_weight_decay_hint": bool(self.no_weight_decay_hint),
            "detail_scope": self.detail_scope,
            "detail_layer": self.detail_layer,
            "detail_kind": self.detail_kind,
            "detail_storage_name": self.detail_storage_name,
        }


@dataclass
class DeliveryCapture:
    """Staged snapshots for one captured optimizer update.

    Everything stays on the parameters' device until ``derive`` runs, which
    happens outside the trainer's timing boundary. Scalar stats are stacked
    per stage so one batched device->host copy per stage suffices instead of
    thousands of per-element syncs.
    """

    step: int
    stages_done: list = field(default_factory=list)
    # roster index -> native-dtype pre-step clone; released in record_post_step
    param_pre_native: dict = field(default_factory=dict)
    scalars: dict = field(default_factory=dict)      # field -> np.float64 (P,)
    meta_int: dict = field(default_factory=dict)     # field -> np.int64 (P,)
    group_float: dict = field(default_factory=dict)  # field -> np.float64 (P,)
    clip_float: dict = field(default_factory=dict)   # field -> float
    clip_int: dict = field(default_factory=dict)     # field -> int
    detail: dict = field(default_factory=dict)       # storage key -> float32 tensor

    def release_snapshots(self) -> None:
        """Drop the temporary all-parameter native clones. Safe to call twice
        and safe to call from an exception path."""
        self.param_pre_native.clear()


# ---------------------------------------------------------------------------
# observer
# ---------------------------------------------------------------------------


class TrainingOptimizerDeliveryObserver:
    """Camera 3: read-only staged observation of the real clip + step event.

    Registers nothing. Calls nothing on the trainer's behalf. Every value is a
    detached clone taken between the trainer's own calls.
    """

    def __init__(self, model: torch.nn.Module, optimizer,
                 capture_steps=None,
                 max_grad_norm: float = 1.0,
                 clip_parameters=None):
        # Defaults bind at import; resolve here so the window can be retargeted.
        capture_steps = CAPTURE_WINDOW if capture_steps is None else capture_steps
        # Reuse Camera 1's constructor wholesale: mixer discovery, per-layer
        # InProjSpec, cross-layer geometry assertion, A_floor presence. No
        # second geometry transcription: headdim / d_state / mimo_rank are
        # never re-derived here.
        base = tto.TimeRetentionObserver(model, capture_steps)
        self.capture_steps = base.capture_steps
        self.layers = base.layers
        self.specs = base.specs
        self.geometry = dict(base.geometry)

        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)

        # Camera 3 attaches nothing to the graph. Kept as an always-empty
        # tuple so the leak assertion is a structural fact, not a hunt.
        self._hook_handles: tuple = ()

        self._build_roster()
        self._build_detail_map()

        clip_params = (
            list(model.parameters()) if clip_parameters is None
            else list(clip_parameters)
        )
        roster_ids = {id(r.param) for r in self.roster}
        clip_ids = {id(p) for p in clip_params}
        missing = roster_ids - clip_ids
        if missing:
            raise ObserverError(
                f"{len(missing)} optimizer parameter(s) are not in the clipped "
                "parameter set; the observed clip event would not cover the "
                "delivered update"
            )
        self.clip_parameters = clip_params
        # Parameters clipped by the trainer but not owned by the optimizer.
        # Empty in the frozen control (AdamW(model.parameters())).
        self._clip_only = [p for p in clip_params if id(p) not in roster_ids]
        self.clip_scope_matches_optimizer_scope = not self._clip_only

        self.geometry["num_optimizer_parameters"] = len(self.roster)
        self.geometry["num_detail_tensors"] = len(self.detail_roster)
        self.geometry["clip_scope_matches_optimizer_scope"] = (
            self.clip_scope_matches_optimizer_scope
        )

    # ------------------------------------------------------------------
    # roster construction
    # ------------------------------------------------------------------

    def _build_roster(self) -> None:
        """Roster by optimizer parameter identity, preserving tied aliases.

        ``remove_duplicate=False`` is what surfaces tied weights (a shared
        embedding / lm_head tensor has two names but is ONE optimizer
        parameter and must not produce two rows).
        """
        aliases: dict = {}
        for name, param in self.model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(param), []).append(name)

        roster: list = []
        seen: dict = {}
        for gi, group in enumerate(self.optimizer.param_groups):
            for offset, param in enumerate(group["params"]):
                pid = id(param)
                if pid in seen:
                    raise ObserverError(
                        f"parameter {aliases.get(pid, ['<unnamed>'])[0]!r} "
                        f"appears twice in the optimizer "
                        f"(groups {seen[pid]} and {gi}); refusing an ambiguous "
                        "delivery roster"
                    )
                seen[pid] = gi
                names = tuple(aliases.get(pid, ()))
                canonical = names[0] if names else f"unnamed_g{gi}_o{offset:04d}"
                roster.append(ParamRecord(
                    index=len(roster),
                    parameter_id=canonical,
                    canonical_name=canonical,
                    aliases=names,
                    group_index=gi,
                    group_offset=offset,
                    shape=tuple(int(s) for s in param.shape),
                    source_dtype=str(param.dtype),
                    numel=int(param.numel()),
                    no_weight_decay_hint=bool(
                        getattr(param, "_no_weight_decay", False)
                    ),
                    detail_scope="scalar_only",
                    param=param,
                ))
        if not roster:
            raise ObserverError("optimizer exposes no parameters to observe")
        ids = [r.parameter_id for r in roster]
        if len(set(ids)) != len(ids):
            raise ObserverError(
                "optimizer parameter ids are not unique; roster identity would "
                "be ambiguous"
            )
        self.roster = roster
        self._by_param_id = {id(r.param): r for r in roster}

    def _build_detail_map(self) -> None:
        """Assign full-vector detail scopes from realized mixer geometry.

        in_proj.weight rows are sliced with ``InProjSpec.bounds`` -- the SAME
        indices Camera 2 applies to the in_proj output, applied here to the
        rows that produce those output features (G7). The bounds are never
        re-derived by hand.
        """
        detail_roster: list = []
        for layer_idx, (layer_name, mixer) in enumerate(self.layers):
            spec = self.specs[layer_idx]
            weight = mixer.in_proj.weight
            if weight.ndim != 2:
                raise GeometryMismatchError(
                    f"layer {layer_idx} ({layer_name}): in_proj.weight is "
                    f"{weight.ndim}-D, expected 2-D (out_features, in_features)"
                )
            if int(weight.shape[0]) != int(spec.total):
                raise GeometryMismatchError(
                    f"layer {layer_idx} ({layer_name}): in_proj.weight rows "
                    f"{int(weight.shape[0])} != spec total {int(spec.total)}"
                )
            record = self._by_param_id.get(id(weight))
            if record is None:
                raise ObserverError(
                    f"layer {layer_idx} ({layer_name}): in_proj.weight is not "
                    "an optimizer parameter; the control-row join to Camera 2 "
                    "would be unobservable"
                )
            record.detail_scope = "row_slice"
            record.detail_layer = layer_idx
            record.detail_kind = "in_proj_weight_rows"
            record.detail_storage_name = "in_proj_weight"
            for region in CONTROL_REGIONS:
                lo, hi = spec.bounds[region]
                detail_roster.append({
                    "layer": layer_idx,
                    "layer_module_path": layer_name,
                    "kind": "row_slice",
                    "storage_name": region,
                    "parameter_id": record.parameter_id,
                    "row_lo": int(lo),
                    "row_hi": int(hi),
                    "shape": [int(hi - lo), int(weight.shape[1])],
                })

            for attr in ("dt_bias", "D"):
                param = getattr(mixer, attr, None)
                if param is None:
                    continue
                rec = self._by_param_id.get(id(param))
                if rec is None:
                    continue
                rec.detail_scope = "full_tensor"
                rec.detail_layer = layer_idx
                rec.detail_kind = "mixer_parameter"
                rec.detail_storage_name = attr
                detail_roster.append({
                    "layer": layer_idx,
                    "layer_module_path": layer_name,
                    "kind": "full_tensor",
                    "storage_name": attr,
                    "parameter_id": rec.parameter_id,
                    "row_lo": None,
                    "row_hi": None,
                    "shape": [int(s) for s in param.shape],
                })

            norm = getattr(mixer, "norm", None)
            norm_weight = getattr(norm, "weight", None) if norm is not None else None
            if norm_weight is not None:
                rec = self._by_param_id.get(id(norm_weight))
                if rec is not None:
                    rec.detail_scope = "full_tensor"
                    rec.detail_layer = layer_idx
                    rec.detail_kind = "mixer_parameter"
                    rec.detail_storage_name = "norm_weight"
                    detail_roster.append({
                        "layer": layer_idx,
                        "layer_module_path": layer_name,
                        "kind": "full_tensor",
                        "storage_name": "norm_weight",
                        "parameter_id": rec.parameter_id,
                        "row_lo": None,
                        "row_hi": None,
                        "shape": [int(s) for s in norm_weight.shape],
                    })

        self.detail_roster = detail_roster
        self._detailed_indices = [
            r.index for r in self.roster if r.detail_scope != "scalar_only"
        ]

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def should_capture(self, step: int) -> bool:
        return int(step) in self.capture_steps

    def hooks_active(self) -> bool:
        """Always False: Camera 3 registers no module or tensor hooks.

        Kept for loop-shape parity with Cameras 1/2, whose trainers assert on
        it every step. Here it is a structural invariant, not a runtime hunt.
        """
        return bool(self._hook_handles)

    def registered_hook_count(self) -> int:
        return len(self._hook_handles)

    def parameter_roster(self) -> list:
        return [r.as_manifest_entry() for r in self.roster]

    def _grad_dtypes(self) -> set:
        return {
            p.grad.dtype for p in self.clip_parameters if p.grad is not None
        }

    def norm_check_rtol(self) -> float:
        """Tolerance for the clip cross-checks, from the coarsest live grad
        dtype in the clip set."""
        dtypes = self._grad_dtypes()
        if not dtypes:
            return _DEFAULT_NORM_RTOL
        return max(norm_rtol_for_dtype(d) for d in dtypes)

    # ------------------------------------------------------------------
    # staged capture
    # ------------------------------------------------------------------

    @contextmanager
    def capture(self, step: int):
        """Bracket the trainer's clip + step. Entered AFTER loss.backward().

        The ``finally`` releases the temporary all-parameter snapshots if an
        exception lands between stages, so a failed step cannot leak a
        model-sized clone into the next one.
        """
        if not self.should_capture(step):
            raise ObserverError(
                f"capture requested for step {step}, outside window "
                f"{sorted(self.capture_steps)}"
            )
        cap = DeliveryCapture(step=int(step))
        try:
            yield cap
        finally:
            cap.release_snapshots()

    def _require_stage(self, cap: DeliveryCapture, stage: str) -> None:
        expected = STAGES[len(cap.stages_done)] if len(cap.stages_done) < len(STAGES) else None
        if stage != expected:
            raise ObserverError(
                f"step {cap.step}: stage {stage!r} recorded out of order; "
                f"stages so far {cap.stages_done}, expected {expected!r}"
            )
        cap.stages_done.append(stage)

    @staticmethod
    def _stack_stage(rows: list, fields: tuple) -> dict:
        """One batched device->host copy for a whole stage's scalars."""
        if not rows:
            return {name: np.zeros((0,), dtype=np.float64) for name in fields}
        block = torch.stack([torch.stack(r) for r in rows])  # (P, F)
        host = block.detach().to("cpu", torch.float64).numpy()
        return {name: host[:, i].copy() for i, name in enumerate(fields)}

    @staticmethod
    def _state_step(state: dict):
        if "step" not in state:
            return None
        value = state["step"]
        return float(value) if torch.is_tensor(value) else float(value)

    def _detail_slices(self, record: ParamRecord, tensor: torch.Tensor,
                       field_name: str, cap: DeliveryCapture) -> None:
        """Store the float32 detail vectors for one parameter and field.

        Row slices use the realized ``InProjSpec.bounds``; a clone is taken so
        the compact slice does not retain the whole weight's storage.
        """
        if record.detail_scope == "scalar_only":
            return
        layer = record.detail_layer
        source = tensor.detach()
        if record.detail_scope == "row_slice":
            spec = self.specs[layer]
            for region in CONTROL_REGIONS:
                lo, hi = spec.bounds[region]
                key = f"layer_{layer:02d}/{region}/{field_name}"
                cap.detail[key] = source[lo:hi].to(torch.float32).clone()
        else:
            key = (
                f"layer_{layer:02d}/{record.detail_storage_name}/{field_name}"
            )
            cap.detail[key] = source.to(torch.float32).clone()

    # -- stage 1: after backward, before clip ---------------------------

    def record_pre_clip(self, cap: DeliveryCapture) -> None:
        """Pre-clip gradients, pre-step parameters and moments.

        Retains a native-dtype pre-step clone of EVERY optimizer parameter --
        without it an all-parameter delta claim is impossible (G17). The
        clones are released in ``record_post_step``.
        """
        self._require_stage(cap, "pre_clip")

        rows: list = []
        meta: dict = {name: [] for name in PARAM_META_INT_FIELDS}
        gfloat: dict = {name: [] for name in PARAM_GROUP_FLOAT_FIELDS}

        for record in self.roster:
            param = record.param
            group = self.optimizer.param_groups[record.group_index]
            state = self.optimizer.state.get(param, {})
            grad = param.grad
            grad_present = grad is not None

            if tuple(param.shape) != record.shape:
                raise GeometryMismatchError(
                    f"step {cap.step}: parameter {record.parameter_id} shape "
                    f"{tuple(param.shape)} != roster shape {record.shape}"
                )

            zero = torch.zeros((), dtype=torch.float32, device=param.device)
            if grad_present:
                if tuple(grad.shape) != record.shape:
                    raise GeometryMismatchError(
                        f"step {cap.step}: {record.parameter_id} gradient shape "
                        f"{tuple(grad.shape)} != parameter shape {record.shape}"
                    )
                g32 = grad.detach().to(torch.float32)
                grad_l2 = torch.linalg.vector_norm(g32)
                grad_max = g32.abs().max()
            else:
                grad_l2 = zero
                grad_max = zero

            p32 = param.detach().to(torch.float32)
            param_l2 = torch.linalg.vector_norm(p32)
            param_max = p32.abs().max()

            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            state_present_pre = exp_avg is not None and exp_avg_sq is not None
            if state_present_pre:
                m32 = exp_avg.detach().to(torch.float32)
                v32 = exp_avg_sq.detach().to(torch.float32)
                m_l2 = torch.linalg.vector_norm(m32)
                v_l2 = torch.linalg.vector_norm(v32)
            else:
                m_l2 = zero
                v_l2 = zero

            if record.detail_scope != "scalar_only":
                if not grad_present:
                    raise ObserverError(
                        f"step {cap.step}: detailed control parameter "
                        f"{record.parameter_id} has no gradient; refusing to "
                        "store an imputed detail basis"
                    )
                if not state_present_pre:
                    raise ObserverError(
                        f"step {cap.step}: detailed control parameter "
                        f"{record.parameter_id} has no pre-step AdamW state; "
                        "the capture window must begin after the parameter's "
                        "first update"
                    )
                self._detail_slices(record, param, "param_pre", cap)
                self._detail_slices(record, grad, "grad_pre_clip", cap)
                self._detail_slices(record, exp_avg, "exp_avg_pre", cap)
                self._detail_slices(record, exp_avg_sq, "exp_avg_sq_pre", cap)

            # Temporary native-dtype pre-step clone; released post-step.
            cap.param_pre_native[record.index] = param.detach().clone()

            rows.append([grad_l2, grad_max, param_l2, param_max, m_l2, v_l2])

            betas = group.get("betas", (0.9, 0.999))
            meta["numel"].append(record.numel)
            meta["group_index"].append(record.group_index)
            meta["group_offset"].append(record.group_offset)
            meta["grad_present"].append(int(grad_present))
            meta["state_present_pre"].append(int(state_present_pre))
            meta["state_present_post"].append(-1)   # filled at post_step
            step_pre = self._state_step(state)
            meta["state_step_pre"].append(
                int(step_pre) if step_pre is not None else -1
            )
            meta["state_step_post"].append(-1)      # filled at post_step
            meta["detail_scope_code"].append(
                DETAIL_SCOPE_CODES[record.detail_scope]
            )
            meta["no_weight_decay_hint"].append(int(record.no_weight_decay_hint))
            meta["amsgrad"].append(int(bool(group.get("amsgrad", False))))
            meta["maximize"].append(int(bool(group.get("maximize", False))))

            gfloat["group_lr"].append(float(group["lr"]))
            gfloat["group_weight_decay"].append(
                float(group.get("weight_decay", 0.0))
            )
            gfloat["group_beta1"].append(float(betas[0]))
            gfloat["group_beta2"].append(float(betas[1]))
            gfloat["group_eps"].append(float(group.get("eps", 1e-8)))
            gfloat["bias_correction1"].append(0.0)  # filled at post_step
            gfloat["bias_correction2"].append(0.0)  # filled at post_step

        cap.scalars.update(self._stack_stage(rows, PRE_CLIP_SCALAR_FIELDS))
        cap.meta_int = {
            name: np.asarray(values, dtype=np.int64)
            for name, values in meta.items()
        }
        cap.group_float = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in gfloat.items()
        }

        # Clip-scope norm: the trainer clips self.clip_parameters, which may be
        # a superset of the optimizer roster. Both scopes are recorded.
        extra_sq = self._extra_grad_sq_sum()
        opt_sq = float(np.sum(np.square(cap.scalars["pre_clip_grad_l2"])))
        cap.clip_float["observer_pre_clip_total_norm_optimizer_scope"] = float(
            np.sqrt(opt_sq)
        )
        cap.clip_float["observer_pre_clip_total_norm"] = float(
            np.sqrt(opt_sq + extra_sq)
        )
        cap.clip_float["clip_max_norm"] = self.max_grad_norm
        cap.clip_int["clip_scope_matches_optimizer_scope"] = int(
            self.clip_scope_matches_optimizer_scope
        )

    def _extra_grad_sq_sum(self) -> float:
        """Squared-norm contribution of clipped-but-not-optimized parameters.

        Zero in the frozen control; computed rather than assumed so the
        recorded clip-scope norm is correct if the scopes ever diverge.
        """
        if not self._clip_only:
            return 0.0
        norms = [
            torch.linalg.vector_norm(p.grad.detach().to(torch.float32))
            for p in self._clip_only if p.grad is not None
        ]
        if not norms:
            return 0.0
        host = torch.stack(norms).detach().to("cpu", torch.float64).numpy()
        return float(np.sum(np.square(host)))

    # -- stage 2: after the trainer's clip_grad_norm_, before step ------

    def record_clip(self, cap: DeliveryCapture, returned_norm) -> None:
        """Post-clip gradients plus the three-view clip cross-check.

        ``returned_norm`` is the value the TRAINER's real ``clip_grad_norm_``
        returned. The observer never calls the clipper itself.
        """
        self._require_stage(cap, "clip")

        rows: list = []
        for record in self.roster:
            param = record.param
            grad = param.grad
            zero = torch.zeros((), dtype=torch.float32, device=param.device)
            if grad is None:
                rows.append([zero, zero])
                continue
            g32 = grad.detach().to(torch.float32)
            rows.append([torch.linalg.vector_norm(g32), g32.abs().max()])
            if record.detail_scope != "scalar_only":
                self._detail_slices(record, grad, "grad_post_clip", cap)

        cap.scalars.update(self._stack_stage(rows, CLIP_SCALAR_FIELDS))

        extra_sq = self._extra_grad_sq_sum()
        opt_sq = float(np.sum(np.square(cap.scalars["post_clip_grad_l2"])))
        post_opt = float(np.sqrt(opt_sq))
        post_total = float(np.sqrt(opt_sq + extra_sq))
        cap.clip_float["post_clip_total_norm_optimizer_scope"] = post_opt
        cap.clip_float["post_clip_total_norm"] = post_total

        pre_total = cap.clip_float["observer_pre_clip_total_norm"]
        returned = float(returned_norm)
        if not np.isfinite(returned):
            raise NonFiniteCaptureError(
                f"step {cap.step}: clip_grad_norm_ returned {returned}"
            )
        cap.clip_float["returned_total_norm"] = returned

        if pre_total == 0.0:
            # An undefined empirical coefficient must not be papered over with
            # a stand-in value. A real training step cannot have an exactly
            # zero global gradient; stopping here is the honest outcome.
            raise ObserverError(
                f"step {cap.step}: observer pre-clip global gradient norm is "
                "exactly zero; the empirical clip coefficient is undefined and "
                "will not be imputed"
            )

        rtol = self.norm_check_rtol()

        # View 1 vs view 2: the observer's float32 pre-clip norm against the
        # norm the trainer's clipper actually returned.
        if not _close(pre_total, returned, rtol):
            raise ObserverError(
                f"step {cap.step}: clip cross-check failed -- observer "
                f"pre-clip total norm {pre_total!r} disagrees with the norm "
                f"returned by clip_grad_norm_ {returned!r} beyond rtol={rtol}"
            )

        expected = clip_coefficient_expected(self.max_grad_norm, returned)
        empirical = post_total / pre_total
        cap.clip_float["clip_coefficient_expected"] = expected
        cap.clip_float["clip_coefficient_empirical"] = empirical
        cap.clip_int["clip_fired"] = int(expected < 1.0)

        # View 3: the empirical global post/pre ratio against the exact
        # expected coefficient. Coordinate-wise ratios are never the estimator.
        if not _close(empirical, expected, rtol):
            raise ObserverError(
                f"step {cap.step}: clip cross-check failed -- empirical "
                f"coefficient {empirical!r} (post/pre global norm ratio) "
                f"disagrees with expected coefficient {expected!r} = "
                f"min(1, {self.max_grad_norm}/(returned+{CLIP_EPS})) beyond "
                f"rtol={rtol}"
            )

        # Detailed vectors: post-clip gradient must be the uniform rescale of
        # the pre-clip gradient by the expected coefficient.
        for key, post in cap.detail.items():
            if not key.endswith("/grad_post_clip"):
                continue
            pre = cap.detail[key[: -len("grad_post_clip")] + "grad_pre_clip"]
            if not torch.allclose(post, pre * expected, rtol=rtol, atol=1e-6):
                raise ObserverError(
                    f"step {cap.step}: detailed post-clip gradient {key} is "
                    f"not the pre-clip gradient scaled by the expected "
                    f"coefficient {expected!r} within rtol={rtol}"
                )

    # -- stage 3: after the trainer's optimizer.step() ------------------

    def record_post_step(self, cap: DeliveryCapture) -> None:
        """Realized delta and the fp32 semantic AdamW decomposition.

        The realized delta is a float32 subtraction of native parameters. The
        idealized terms use the ACTUAL group settings and the observed post-step
        state; the difference is a realization residual that includes native
        optimizer arithmetic and backend rounding as well as parameter
        rounding. It is not called quantization error, and no global effective
        learning rate is produced.
        """
        self._require_stage(cap, "post_step")

        rows: list = []
        for record in self.roster:
            param = record.param
            group = self.optimizer.param_groups[record.group_index]
            state = self.optimizer.state.get(param, {})
            index = record.index

            pre_native = cap.param_pre_native.get(index)
            if pre_native is None:
                raise ObserverError(
                    f"step {cap.step}: no pre-step snapshot for "
                    f"{record.parameter_id}; an all-parameter delta cannot be "
                    "claimed from a subset"
                )

            device = param.device
            zero = torch.zeros((), dtype=torch.float32, device=device)

            p_post32 = param.detach().to(torch.float32)
            p_pre32 = pre_native.to(torch.float32)
            delta = p_post32 - p_pre32          # DELTA_PRECISION, exactly

            param_l2 = torch.linalg.vector_norm(p_post32)
            param_max = p_post32.abs().max()
            delta_l2 = torch.linalg.vector_norm(delta)
            delta_max = delta.abs().max()
            # bf16 floor: below-resolution deltas land at exactly zero and are
            # RECORDED, never imputed or denoised (G10).
            delta_zero_fraction = (delta == 0).to(torch.float32).mean()

            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            state_present_post = exp_avg is not None and exp_avg_sq is not None
            grad_present = bool(cap.meta_int["grad_present"][index])

            step_post = self._state_step(state)
            beta1 = float(group.get("betas", (0.9, 0.999))[0])
            beta2 = float(group.get("betas", (0.9, 0.999))[1])
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            eps = float(group.get("eps", 1e-8))
            amsgrad = bool(group.get("amsgrad", False))

            if state_present_post and step_post is not None and step_post > 0:
                bias_correction1 = 1.0 - beta1 ** step_post
                bias_correction2 = 1.0 - beta2 ** step_post
            else:
                bias_correction1 = 0.0
                bias_correction2 = 0.0

            if grad_present and state_present_post and bias_correction1 > 0.0:
                m32 = exp_avg.detach().to(torch.float32)
                v32 = exp_avg_sq.detach().to(torch.float32)
                if amsgrad and state.get("max_exp_avg_sq") is not None:
                    v_used = state["max_exp_avg_sq"].detach().to(torch.float32)
                else:
                    v_used = v32
                m_l2 = torch.linalg.vector_norm(m32)
                v_l2 = torch.linalg.vector_norm(v32)
                # Decoupled decay, as AdamW applies it to the PRE-step value:
                #   p <- p*(1 - lr*wd)  =>  delta_decay = -lr*wd*p_pre
                ideal_decay = p_pre32 * (-(lr * weight_decay))
                denom = (
                    v_used.sqrt() / float(np.sqrt(bias_correction2))
                ) + eps
                ideal_adaptive = m32 / denom * (-(lr / bias_correction1))
                ideal_total = ideal_decay + ideal_adaptive
                residual = delta - ideal_total
                decay_l2 = torch.linalg.vector_norm(ideal_decay)
                adaptive_l2 = torch.linalg.vector_norm(ideal_adaptive)
                total_l2 = torch.linalg.vector_norm(ideal_total)
                residual_l2 = torch.linalg.vector_norm(residual)
                residual_max = residual.abs().max()
                inner = torch.sum(delta * ideal_total)
            else:
                # AdamW skips a parameter whose grad is None -- INCLUDING its
                # decoupled decay. Zero here is the optimizer's definition of
                # "not delivered", not a stand-in for an unmeasured value.
                m_l2 = zero
                v_l2 = zero
                decay_l2 = zero
                adaptive_l2 = zero
                total_l2 = zero
                residual_l2 = delta_l2
                residual_max = delta_max
                inner = zero
                if not grad_present and float(delta_max) != 0.0:
                    raise ObserverError(
                        f"step {cap.step}: {record.parameter_id} has no "
                        "gradient (AdamW skips it entirely, decay included) "
                        f"yet its parameters moved by up to "
                        f"{float(delta_max)!r}; something outside the observed "
                        "optimizer delivery changed this parameter"
                    )

            if record.detail_scope != "scalar_only":
                if not state_present_post:
                    raise ObserverError(
                        f"step {cap.step}: detailed control parameter "
                        f"{record.parameter_id} has no post-step AdamW state"
                    )
                self._detail_slices(record, exp_avg, "exp_avg_post", cap)
                self._detail_slices(record, exp_avg_sq, "exp_avg_sq_post", cap)
                self._detail_slices(record, delta, "delta", cap)

            rows.append([
                param_l2, param_max, m_l2, v_l2,
                delta_l2, delta_max, delta_zero_fraction,
                decay_l2, adaptive_l2, total_l2,
                residual_l2, residual_max, inner,
            ])

            cap.meta_int["state_present_post"][index] = int(state_present_post)
            cap.meta_int["state_step_post"][index] = (
                int(step_post) if step_post is not None else -1
            )
            cap.group_float["bias_correction1"][index] = bias_correction1
            cap.group_float["bias_correction2"][index] = bias_correction2

            # Release this parameter's snapshot and float32 working set as soon
            # as its scalars are stacked, so the peak is one parameter's
            # temporaries plus a shrinking model copy rather than both in full.
            # The embedding alone is ~98M elements on the real checkpoint.
            cap.param_pre_native.pop(index, None)
            del pre_native, p_post32, p_pre32, delta

        cap.scalars.update(self._stack_stage(rows, POST_STEP_SCALAR_FIELDS))
        # Model-sized temporaries die here, every step.
        cap.release_snapshots()

    # ------------------------------------------------------------------
    # completeness + derivation
    # ------------------------------------------------------------------

    def assert_complete(self, cap: DeliveryCapture) -> None:
        """In-memory completeness gate, run after the capture context exits."""
        if cap.stages_done != list(STAGES):
            raise ObserverError(
                f"step {cap.step}: stages recorded {cap.stages_done} != "
                f"required {list(STAGES)}"
            )
        n = len(self.roster)
        for name in PARAM_SCALAR_FIELDS:
            if name not in cap.scalars:
                raise ObserverError(f"step {cap.step}: missing scalar {name}")
            if cap.scalars[name].shape != (n,):
                raise ObserverError(
                    f"step {cap.step}: scalar {name} has shape "
                    f"{cap.scalars[name].shape}, expected ({n},)"
                )
        for name in PARAM_META_INT_FIELDS:
            if cap.meta_int.get(name) is None or cap.meta_int[name].shape != (n,):
                raise ObserverError(
                    f"step {cap.step}: parameter meta {name} missing or "
                    "wrong length"
                )
        if (cap.meta_int["state_present_post"] < 0).any():
            raise ObserverError(
                f"step {cap.step}: post-step state presence never recorded for "
                "every optimizer parameter"
            )
        for name in CLIP_GLOBAL_FLOAT_FIELDS:
            if name not in cap.clip_float:
                raise ObserverError(
                    f"step {cap.step}: missing clip quantity {name}"
                )
        for name in CLIP_GLOBAL_INT_FIELDS:
            if name not in cap.clip_int:
                raise ObserverError(
                    f"step {cap.step}: missing clip flag {name}"
                )
        expected_detail = {
            f"layer_{d['layer']:02d}/{d['storage_name']}/{f}"
            for d in self.detail_roster for f in DETAIL_FIELDS
        }
        if set(cap.detail) != expected_detail:
            missing = sorted(expected_detail - set(cap.detail))
            extra = sorted(set(cap.detail) - expected_detail)
            raise ObserverError(
                f"step {cap.step}: detail tensors incomplete; missing "
                f"{missing[:4]} extra {extra[:4]}"
            )
        if cap.param_pre_native:
            raise ObserverError(
                f"step {cap.step}: temporary pre-step parameter snapshots were "
                "not released"
            )

    def derive(self, cap: DeliveryCapture, token_ids, *, training_loss: float,
               dataset_row: int, optimizer_step: int) -> tto.DerivedCapture:
        """Package the staged capture into a DerivedCapture.

        Raises NonFiniteCaptureError on ANY non-finite scalar, detail tensor,
        clip quantity, or loss. Stored values are the observed ones; nothing is
        rescaled, smoothed, or filled in.
        """
        self.assert_complete(cap)

        if not np.isfinite(training_loss):
            raise NonFiniteCaptureError(
                f"step {cap.step}: non-finite training loss {training_loss}"
            )

        arrays: dict = {}
        records: list = []
        raw_bytes = 0

        def _record(key, name, arr, source_dtype, storage_dtype, layer=None,
                    region=None, numeric=True):
            nonlocal raw_bytes
            arrays[key] = arr
            raw_bytes += arr.nbytes
            total = int(arr.size)
            if numeric:
                finite = int(np.count_nonzero(np.isfinite(arr)))
            else:
                finite = total
            records.append({
                "key": key,
                "layer": layer,
                "name": name,
                "region": region,
                "shape": list(arr.shape),
                "source_dtype": source_dtype,
                "storage_dtype": storage_dtype,
                "finite_count": finite,
                "nonfinite_count": total - finite,
                "min": float(np.min(arr)) if (numeric and total) else None,
                "max": float(np.max(arr)) if (numeric and total) else None,
                "mean": float(np.mean(arr)) if (numeric and total) else None,
                "std": float(np.std(arr)) if (numeric and total > 1) else (
                    0.0 if numeric else None
                ),
                "raw_bytes": int(arr.nbytes),
            })

        def _admit_numeric(key, name, arr, source_dtype, storage_dtype,
                           layer=None, region=None):
            if not np.all(np.isfinite(arr)):
                bad = int(arr.size - np.count_nonzero(np.isfinite(arr)))
                raise NonFiniteCaptureError(
                    f"step {cap.step} array {key}: {bad} non-finite of "
                    f"{int(arr.size)}"
                )
            _record(key, name, arr, source_dtype, storage_dtype, layer, region)

        # -- detail tensors (one device->host copy each, outside the loop's
        #    timing boundary; the trainer calls derive after step_time froze).
        detail_by_key = {
            f"layer_{d['layer']:02d}/{d['storage_name']}/{f}": (d, f)
            for d in self.detail_roster for f in DETAIL_FIELDS
        }
        for key in sorted(cap.detail):
            entry, field_name = detail_by_key[key]
            tensor = cap.detail[key].detach().to("cpu", torch.float32)
            arr = tensor.numpy()
            expected_shape = tuple(entry["shape"])
            if tuple(arr.shape) != expected_shape:
                raise GeometryMismatchError(
                    f"step {cap.step}: detail {key} shape {tuple(arr.shape)} "
                    f"!= roster shape {expected_shape}"
                )
            _admit_numeric(
                key, field_name, arr,
                "float32" if field_name == "delta" else "native",
                "float32",
                layer=entry["layer"], region=entry["storage_name"],
            )

        # -- per-parameter scalars, indexed by roster order
        for name in PARAM_SCALAR_FIELDS:
            _admit_numeric(
                f"param/{name}", name,
                np.asarray(cap.scalars[name], dtype=np.float64),
                SCALAR_STAT_PRECISION, "float64",
            )
        for name in PARAM_GROUP_FLOAT_FIELDS:
            _admit_numeric(
                f"param/{name}", name,
                np.asarray(cap.group_float[name], dtype=np.float64),
                "python_float", "float64",
            )
        for name in PARAM_META_INT_FIELDS:
            _record(
                f"param_meta/{name}", name,
                np.asarray(cap.meta_int[name], dtype=np.int64),
                "python_int", "int64",
            )

        # -- clip event
        for name in CLIP_GLOBAL_FLOAT_FIELDS:
            _admit_numeric(
                f"clip/{name}", name,
                np.asarray(cap.clip_float[name], dtype=np.float64),
                "python_float", "float64",
            )
        for name in CLIP_GLOBAL_INT_FIELDS:
            _record(
                f"clip/{name}", name,
                np.asarray(cap.clip_int[name], dtype=np.int64),
                "python_int", "int64",
            )

        # -- identity globals
        ids = [int(i) for i in token_ids]
        supervised_token_count = len(ids) - 1  # targets = ids[1:]
        _record("token_ids", "token_ids", np.asarray(ids, dtype="<i8"),
                "int64", "int64")
        _admit_numeric("training_loss", "training_loss",
                       np.float32(training_loss).reshape(()),
                       "float32", "float32")
        _record("supervised_token_count", "supervised_token_count",
                np.asarray(supervised_token_count, dtype="<i8"),
                "int64", "int64")
        _record("dataset_row", "dataset_row",
                np.asarray(int(dataset_row), dtype="<i8"), "int64", "int64")
        _record("optimizer_step", "optimizer_step",
                np.asarray(int(optimizer_step), dtype="<i8"), "int64", "int64")
        _record("parameter_count", "parameter_count",
                np.asarray(len(self.roster), dtype="<i8"), "int64", "int64")
        # Self-describing roster order: guards against roster/array
        # misalignment without needing the manifest.
        names = np.asarray([r.parameter_id for r in self.roster], dtype=np.str_)
        _record("parameter_names", "parameter_names", names,
                "str", str(names.dtype), numeric=False)

        return tto.DerivedCapture(
            step=cap.step,
            phase=PHASE,
            arrays=arrays,
            tensor_records=records,
            token_ids=ids,
            token_sha256=tto.token_ids_sha256(ids),
            sequence_length=supervised_token_count,
            raw_bytes=raw_bytes,
        )


# ---------------------------------------------------------------------------
# compact W&B scalar summaries (no tensor arrays in history)
# ---------------------------------------------------------------------------


def delivery_scalars(derived: tto.DerivedCapture) -> dict:
    """The delivery-step scalars merged into the single wandb.log per step."""
    a = derived.arrays

    def _global_l2(name):
        return float(np.sqrt(np.sum(np.square(a[f"param/{name}"]))))

    delta_l2 = a["param/delta_l2"]
    numel = a["param_meta/numel"].astype(np.float64)
    zero_fraction = a["param/delta_fraction_zero"]
    total_numel = float(np.sum(numel))
    return {
        "delivery/pre_clip_total_norm": float(
            a["clip/observer_pre_clip_total_norm"]
        ),
        "delivery/clip_coefficient_expected": float(
            a["clip/clip_coefficient_expected"]
        ),
        "delivery/clip_coefficient_empirical": float(
            a["clip/clip_coefficient_empirical"]
        ),
        "delivery/clip_fired": float(a["clip/clip_fired"]),
        "delivery/delta_l2_global": float(
            np.sqrt(np.sum(np.square(delta_l2)))
        ),
        # Element-weighted so it is the true fraction of parameter elements
        # that did not move, not an unweighted mean over parameters.
        "delivery/delta_fraction_zero_global": (
            float(np.sum(zero_fraction * numel) / total_numel)
            if total_numel else 0.0
        ),
        "delivery/exp_avg_l2_global": _global_l2("exp_avg_post_l2"),
        "delivery/exp_avg_sq_l2_global": _global_l2("exp_avg_sq_post_l2"),
        "delivery/realization_residual_l2_global": _global_l2(
            "realization_residual_l2"
        ),
        "supervised_token_count": int(a["supervised_token_count"]),
    }


# ---------------------------------------------------------------------------
# artifact writing -- acyclic finalization order, identical to the siblings:
#   (a) NPZs  (b) storage_report.json  (c) hash NPZs + storage report
#   (d) manifest.json embedding those hashes  (e) SHA256SUMS covering
#       NPZs + storage_report.json + manifest.json (never itself)
# ---------------------------------------------------------------------------


def expected_filenames() -> list:
    return [
        f"step_{step:06d}_{PHASE}.npz"
        for step in range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1)
    ]


def finalize_optimizer_delivery_dir(out_dir, file_records: list, run_meta: dict,
                                    expected_files: int = 6) -> dict:
    out_dir = Path(out_dir)
    if len(file_records) != expected_files:
        raise ObserverError(
            f"expected {expected_files} capture files, have {len(file_records)}"
        )
    file_records = sorted(file_records, key=lambda r: r["filename"])
    if [r["filename"] for r in file_records] != expected_filenames():
        raise ObserverError(
            f"capture filenames {[r['filename'] for r in file_records]} != "
            f"expected {expected_filenames()}"
        )

    # (b) storage report from NPZ metadata only, with per-quantity ranking.
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
                        "region": t.get("region"),
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
    storage_path.write_text(
        json.dumps(storage_report, sort_keys=True, indent=2) + "\n"
    )

    # (c) hash NPZs and the storage report.
    hashes = {r["filename"]: sha256_file(out_dir / r["filename"]) for r in file_records}
    hashes["storage_report.json"] = sha256_file(storage_path)

    # (d) manifest embeds those hashes; never its own.
    manifest = {
        "optimizer_delivery_schema_version": OPTIMIZER_DELIVERY_SCHEMA_VERSION,
        "camera_number": CAMERA_NUMBER,
        "camera_family": CAMERA_FAMILY,
        "observation_stage": OBSERVATION_STAGE,
        "gradient_stage_split": list(GRADIENT_STAGE_SPLIT),
        "delta_precision": DELTA_PRECISION,
        "clip_coefficient_semantics": CLIP_COEFFICIENT_SEMANTICS,
        "clip_coefficient_expected_rule": (
            "min(1, max_norm / (returned_total_norm + 1e-6)); the empirical "
            "coefficient is the global post/pre gradient-norm ratio; "
            "coordinate-wise ratios are never used as the estimator"
        ),
        "adamw_decomposition_semantics": ADAMW_DECOMPOSITION_SEMANTICS,
        "realization_residual_semantics": (
            "realized_delta - ideal_total_delta; includes native optimizer "
            "arithmetic and backend rounding as well as parameter rounding; "
            "it is NOT pure bf16 quantization error"
        ),
        "effective_learning_rate_reported": False,
        "scalar_stat_precision": SCALAR_STAT_PRECISION,
        "hooks_registered": 0,
        "observer_calls_clip_or_step": False,
        "control_regions": list(CONTROL_REGIONS),
        "control_row_axis": (
            "InProjSpec.bounds applied to in_proj.weight ROWS (output "
            "features), the rows that produce the regions Camera 2 slices "
            "from the in_proj OUTPUT"
        ),
        "camera2_join_semantics": (
            "branch identity by step/layer/control family only; Camera 2 is "
            "token x output-feature and Camera 3 is output-feature x "
            "input-feature, so there is no coordinate-wise join"
        ),
        "a_log_parameter_present": False,
        "excluded_from_v1": [
            "embeddings and LM head (Camera 5 embedding_learning)",
            "full vectors for non-control parameters (scalar stats kept)",
            "full out_proj.weight (deferred; v1 control join is in_proj side)",
        ],
        "detail_fields": list(DETAIL_FIELDS),
        "param_scalar_fields": list(PARAM_SCALAR_FIELDS),
        "param_group_float_fields": list(PARAM_GROUP_FLOAT_FIELDS),
        "param_meta_int_fields": list(PARAM_META_INT_FIELDS),
        "clip_global_float_fields": list(CLIP_GLOBAL_FLOAT_FIELDS),
        "clip_global_int_fields": list(CLIP_GLOBAL_INT_FIELDS),
        "capture_window": sorted({r["step"] for r in file_records}),
        "phases": [PHASE],
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

    # (e) SHA256SUMS covers NPZs + storage_report.json + manifest.json.
    hashes["manifest.json"] = sha256_file(manifest_path)
    lines = [f"{hashes[name]}  {name}" for name in sorted(hashes)]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    return manifest
