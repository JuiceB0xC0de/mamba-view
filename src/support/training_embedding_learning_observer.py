"""Training embedding-learning observer for Mamba-3 (Camera 5, steps 20-25).

Modal-free by design so every mechanism is CPU-testable. The Modal trainer
(modal_embedding_learning_observer.py) imports this module; nothing here
imports modal, wandb, or CUDA-only code.

Camera 5 asks which vocabulary symbols were rewritten by each optimizer update.
On the pinned SISO checkpoint ``backbone.embedding.weight`` and
``lm_head.weight`` are aliases of ONE tied bf16 parameter of realized shape
[128256, 768], so every row has two computational roles: it is looked up when
its token appears in ``input_ids``, and it is the output direction for that
vocabulary item in the LM head.

Lossless native state chain -- the central invariant
----------------------------------------------------
The complete pre-update-20 bf16 bit table plus six sparse native-bit patches
reconstruct every captured post-step table state EXACTLY, bit for bit::

    table = pre_window_bf16_bits                      (step 20 only)
    for each step: table[changed_flat_index] = changed_post_bf16_bits

A coordinate enters a patch iff its native bf16 bits changed -- integer
inequality on the full table, never a float epsilon, never a threshold, never a
top-k filter. Float32 deltas are derived offline by expanding each bf16 bit
pattern exactly into float32 and subtracting. A zero is a real native
measurement floor, not an imputed small update.

The exact observation is the native bf16 table state, not an fp32 idealization
of the AdamW update. Camera 3 already reports scalar delivery statistics for
this same parameter; Camera 5 does not replace that and does not duplicate it.
Camera 5 is row- and coordinate-resolved and records which token roles those
rows occupied.

Read-only with respect to the optimization process
--------------------------------------------------
Camera 5 installs two temporary module forward hooks on captured steps only.
The embedding hook attaches a tensor gradient hook to the ACTUAL embedding
output; that gradient hook clones the incoming native gradient and returns
``None`` -- the gradient stream is never replaced or edited. The LM-head hook
clones ``args[0]``, the actual pre-update hidden tensor. The observer never
calls ``backward()``, ``clip_grad_norm_()``, or ``optimizer.step()``, and
performs no replay and no extra forward.

Two narrowly defined readings, and their limits
-----------------------------------------------
* **embedding-side rewrite**: the exact change to rows used by current input
  tokens.
* **vocabulary-side rewrite**: the exact change to every output row, plus its
  direct frozen-hidden logit displacement ``h_pre @ delta_row``.

The frozen-hidden displacement isolates the local contribution of the tied
table while holding the pre-update final hidden state FIXED. It is not the full
post-update logit change: every other parameter also moves and the hidden
trajectory itself may change. Full causal logit attribution is a later camera.

``output_residual_grad = total_tied_parameter_grad - reconstructed_input_grad``
is an explicitly ALGEBRAIC residual, not an independently captured LM-head
branch gradient. Native CUDA accumulation and the float32 reconstruction can
differ at rounding scale, so the actual embedding-output gradient bits are
stored and the reconstruction can be repeated offline.

No result here may be called memorization, semantic learning, concept
acquisition, or causal attribution. This is a mundane GSM8K calibration window.
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

# Canonical module identities are established by this import; Camera 5 reuses
# Camera 1's error types, deterministic NPZ writer, and token hashing.
import training_time_observer as tto  # noqa: E402

ObserverError = tto.ObserverError
NonFiniteCaptureError = tto.NonFiniteCaptureError
GeometryMismatchError = tto.GeometryMismatchError

write_step_npz = tto.write_step_npz
sha256_file = tto.sha256_file
token_ids_sha256 = tto.token_ids_sha256

EMBEDDING_LEARNING_SCHEMA_VERSION = 1
CAMERA_NUMBER = 5
CAMERA_FAMILY = "embedding_learning"
PHASE = "embedding_learning"
CAPTURE_START_STEP = 20
CAPTURE_END_STEP = 25
CAPTURE_WINDOW = frozenset(range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1))
DETAIL_TOP_K_DELTA = 256
DETAIL_TOP_K_GRAD = 256

OBSERVATION_STAGE = "embedding_learning"
STATE_ENCODING = "native_bf16_bits_baseline_plus_lossless_sparse_patches"
PATCH_MEMBERSHIP = "native_bf16_bit_inequality_over_full_table"
DELTA_PRECISION = "exact_bf16_bit_expansion_to_float32_then_subtract"
RESIDUAL_SEMANTICS = "algebraic_total_minus_reconstructed_input_path"
DISPLACEMENT_SEMANTICS = "frozen_pre_update_hidden_direct_row_contribution"

STAGES = ("pre_clip", "clip", "post_step")

# Token role codes. Derived from the ACTUAL trainer tensors, never hard-coded
# per token: BOS is normally input_only and EOS normally target_only, but those
# are consequences to validate, not labels to assume.
ROLE_NEITHER = 0
ROLE_INPUT_ONLY = 1
ROLE_TARGET_ONLY = 2
ROLE_BOTH = 3
ROLE_NAMES = {
    ROLE_NEITHER: "neither",
    ROLE_INPUT_ONLY: "input_only",
    ROLE_TARGET_ONLY: "target_only",
    ROLE_BOTH: "both",
}

# All-vocabulary row metrics, shape [V]. float32 storage is declared in the
# manifest; the analyzer re-derives the native delta metrics from the bit
# patches independently and compares before reporting anything.
ROW_METRIC_FIELDS = (
    "pre_value_rms",
    "pre_value_l2",
    "pre_clip_grad_l2",
    "pre_clip_grad_max_abs",
    "post_clip_grad_l2",
    "post_clip_grad_max_abs",
    "input_path_grad_l2",
    "output_residual_grad_l2",
    "delta_l2",
    "delta_max_abs",
    "relative_delta_l2",
    "delta_grad_inner",
    "delta_grad_cosine",
    "delta_input_grad_inner",
    "delta_input_grad_cosine",
    "delta_residual_inner",
    "delta_residual_cosine",
)
# Fields whose schema explicitly permits masked NaN for undefined values. The
# finite-only check must not fire on these; their paired *_valid mask is the
# authority. Turning an undefined cosine into 0.0 would assert orthogonality.
ROW_METRIC_MAY_BE_NAN = frozenset({
    "relative_delta_l2",
    "delta_grad_cosine",
    "delta_input_grad_cosine",
    "delta_residual_cosine",
})
ROW_VALIDITY_FIELDS = (
    "relative_delta_l2_valid",
    "delta_grad_cosine_valid",
    "delta_input_grad_cosine_valid",
    "delta_residual_cosine_valid",
)
ROW_COUNT_FIELDS = (
    "changed_coordinate_count",
    "input_count",
    "target_count",
    "role_code",
    "changed_flag",
)


# ---------------------------------------------------------------------------
# native bf16 bit helpers
# ---------------------------------------------------------------------------


def bf16_to_bits(tensor: torch.Tensor) -> np.ndarray:
    """Copy a bf16 tensor's EXACT native bit pattern into a numpy uint16 array.

    The bf16 storage is viewed through a 16-bit integer dtype and reinterpreted
    on the numpy side. The value is never converted to float first, so signed
    zero, subnormals, and every rounding artifact survive untouched.
    """
    if tensor.dtype != torch.bfloat16:
        raise ObserverError(
            f"bf16_to_bits requires a bfloat16 tensor, got {tensor.dtype}; "
            "converting to float before reading bits would destroy the exact "
            "native state"
        )
    view = tensor.detach().contiguous().view(torch.int16).cpu().numpy()
    return view.view(np.uint16).copy()


def bits_to_float32(bits) -> np.ndarray:
    """Expand bf16 bit patterns EXACTLY into float32.

    bf16 is the high 16 bits of an IEEE-754 float32, so shifting left by 16 and
    reinterpreting is lossless in both directions.
    """
    arr = np.asarray(bits)
    if arr.dtype != np.uint16:
        raise ObserverError(
            f"bits_to_float32 requires uint16 bit patterns, got {arr.dtype}"
        )
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)


def rank_descending_stable(values: np.ndarray, k: int) -> np.ndarray:
    """Top-k indices by DESCENDING value, ties broken by ASCENDING index.

    A device ``topk`` is not a stable identity authority when values tie, and
    detail-row membership must be reproducible. lexsort's last key is primary.
    """
    values = np.asarray(values, dtype=np.float64)
    if k <= 0 or values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.lexsort((np.arange(values.size, dtype=np.int64), -values))
    return order[: min(int(k), values.size)].astype(np.int64)


# ---------------------------------------------------------------------------
# tied-parameter discovery
# ---------------------------------------------------------------------------


@dataclass
class TiedTable:
    """The single tied embedding / LM-head parameter, discovered live."""

    param: torch.nn.Parameter
    canonical_name: str
    aliases: tuple
    embedding_module: torch.nn.Module
    lm_head_module: torch.nn.Module
    embedding_path: str
    lm_head_path: str
    group_index: int
    group_offset: int
    vocab_size: int
    hidden_size: int
    source_dtype: str

    def as_manifest_entry(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "embedding_module_path": self.embedding_path,
            "lm_head_module_path": self.lm_head_path,
            "group_index": self.group_index,
            "group_offset": self.group_offset,
            "shape": [self.vocab_size, self.hidden_size],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "source_dtype": self.source_dtype,
            "tied": True,
        }


def _resolve_module(model: torch.nn.Module, path: str):
    obj = model
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def discover_tied_table(model: torch.nn.Module, optimizer) -> TiedTable:
    """Identity-based discovery of the tied embedding / LM-head table.

    Fails loudly on ambiguity. This is a contract for the pinned model, not a
    generic embedding inspector: there is deliberately no fallback to "the
    first two-dimensional parameter".
    """
    embedding_path = None
    for candidate in ("backbone.embedding", "embedding", "backbone.embeddings"):
        module = _resolve_module(model, candidate)
        if module is not None and getattr(module, "weight", None) is not None:
            embedding_path = candidate
            break
    if embedding_path is None:
        raise ObserverError(
            "no token-lookup module found: expected backbone.embedding with a "
            ".weight parameter"
        )
    lm_head_path = None
    for candidate in ("lm_head", "backbone.lm_head"):
        module = _resolve_module(model, candidate)
        if module is not None and getattr(module, "weight", None) is not None:
            lm_head_path = candidate
            break
    if lm_head_path is None:
        raise ObserverError(
            "no vocabulary-projection module found: expected lm_head with a "
            ".weight parameter"
        )

    embedding_module = _resolve_module(model, embedding_path)
    lm_head_module = _resolve_module(model, lm_head_path)
    weight = embedding_module.weight
    if lm_head_module.weight is not weight:
        raise ObserverError(
            f"{embedding_path}.weight and {lm_head_path}.weight are NOT the "
            "same Parameter object; Camera 5 v1 is a contract for the pinned "
            "tied checkpoint and refuses to treat an untied table as tied"
        )

    aliases = tuple(sorted(
        name for name, param in model.named_parameters(remove_duplicate=False)
        if param is weight
    ))
    if not aliases:
        raise ObserverError("tied table is not a registered model parameter")

    matches = [
        (gi, offset)
        for gi, group in enumerate(optimizer.param_groups)
        for offset, param in enumerate(group["params"])
        if param is weight
    ]
    if len(matches) != 1:
        raise ObserverError(
            f"tied table appears {len(matches)} times across optimizer groups "
            f"(expected exactly 1): {matches}"
        )
    group_index, group_offset = matches[0]

    if weight.ndim != 2:
        raise GeometryMismatchError(
            f"tied table is {weight.ndim}-D, expected 2-D (vocab, hidden)"
        )
    if weight.dtype != torch.bfloat16:
        raise ObserverError(
            f"tied table dtype is {weight.dtype}; Camera 5 v1 records exact "
            "native bf16 bit patterns and refuses a non-bf16 source tensor"
        )

    canonical = (
        f"{embedding_path}.weight"
        if f"{embedding_path}.weight" in aliases else aliases[0]
    )
    return TiedTable(
        param=weight,
        canonical_name=canonical,
        aliases=aliases,
        embedding_module=embedding_module,
        lm_head_module=lm_head_module,
        embedding_path=embedding_path,
        lm_head_path=lm_head_path,
        group_index=group_index,
        group_offset=group_offset,
        vocab_size=int(weight.shape[0]),
        hidden_size=int(weight.shape[1]),
        source_dtype=str(weight.dtype),
    )


# ---------------------------------------------------------------------------
# capture container
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingCapture:
    """One-step staged capture. Never holds six full tables."""

    step: int
    input_ids: np.ndarray
    targets: np.ndarray
    stages_done: list = field(default_factory=list)

    # hook products (native bits, cloned from the ACTUAL live tensors)
    embedding_output_grad: torch.Tensor | None = None
    lm_head_input: torch.Tensor | None = None
    embedding_forward_count: int = 0
    lm_head_forward_count: int = 0
    embedding_grad_count: int = 0

    # temporary native snapshots, released in record_post_step
    table_pre: torch.Tensor | None = None
    grad_pre_clip: torch.Tensor | None = None
    grad_post_clip: torch.Tensor | None = None

    baseline_bits: np.ndarray | None = None      # step 20 only
    row_metrics: dict = field(default_factory=dict)
    row_counts: dict = field(default_factory=dict)
    patch: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)
    scalars: dict = field(default_factory=dict)
    returned_grad_norm: float | None = None
    learning_rate: float | None = None

    def release_snapshots(self) -> None:
        """Drop every model-sized temporary. Safe to call twice and safe from
        an exception path."""
        self.table_pre = None
        self.grad_pre_clip = None
        self.grad_post_clip = None


# ---------------------------------------------------------------------------
# observer
# ---------------------------------------------------------------------------


class TrainingEmbeddingLearningObserver:
    """Camera 5: lossless native recorder of the tied embedding / LM-head table."""

    def __init__(self, model: torch.nn.Module, optimizer,
                 capture_steps=None):
        # Defaults bind at import; resolve here so the window can be retargeted.
        capture_steps = CAPTURE_WINDOW if capture_steps is None else capture_steps
        self.capture_steps = frozenset(int(s) for s in capture_steps)
        self.model = model
        self.optimizer = optimizer
        self.table = discover_tied_table(model, optimizer)
        self._handles: list = []
        self.geometry = {
            "vocab_size": self.table.vocab_size,
            "hidden_size": self.table.hidden_size,
            "tied_parameter": self.table.canonical_name,
            "tied_aliases": list(self.table.aliases),
            "embedding_module_path": self.table.embedding_path,
            "lm_head_module_path": self.table.lm_head_path,
            "source_dtype": self.table.source_dtype,
        }

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def should_capture(self, step: int) -> bool:
        return int(step) in self.capture_steps

    def hooks_active(self) -> bool:
        return bool(self._handles)

    def _remove_hooks(self) -> None:
        """Idempotent: called after backward and again from every finally."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------
    # capture lifecycle
    # ------------------------------------------------------------------

    def _make_embedding_hook(self, cap: EmbeddingCapture):
        expected_positions = int(cap.input_ids.size)

        def hook(module, args, output):
            cap.embedding_forward_count += 1
            if cap.embedding_forward_count > 1:
                raise ObserverError(
                    f"step {cap.step}: embedding forward hook fired "
                    f"{cap.embedding_forward_count} times; exactly one forward "
                    "per captured step is required"
                )
            if not isinstance(output, torch.Tensor):
                raise ObserverError(
                    f"step {cap.step}: embedding output is not a tensor "
                    f"({type(output).__name__})"
                )
            if output.ndim != 3 or output.shape[0] != 1:
                raise GeometryMismatchError(
                    f"step {cap.step}: embedding output shape "
                    f"{tuple(output.shape)}, expected (1, T, D)"
                )
            if int(output.shape[1]) != expected_positions:
                raise GeometryMismatchError(
                    f"step {cap.step}: embedding output has "
                    f"{int(output.shape[1])} positions but input_ids has "
                    f"{expected_positions}"
                )
            if int(output.shape[2]) != self.table.hidden_size:
                raise GeometryMismatchError(
                    f"step {cap.step}: embedding output width "
                    f"{int(output.shape[2])} != hidden size "
                    f"{self.table.hidden_size}"
                )
            if not output.requires_grad:
                raise ObserverError(
                    f"step {cap.step}: embedding output does not require grad; "
                    "Camera 5 must observe the real training forward"
                )

            def grad_hook(grad):
                cap.embedding_grad_count += 1
                if cap.embedding_grad_count > 1:
                    raise ObserverError(
                        f"step {cap.step}: embedding gradient hook fired "
                        "twice; unexpected second backward delivery"
                    )
                cap.embedding_output_grad = grad.detach().clone()
                # Mandatory: never replace or alter the gradient.
                return None

            self._handles.append(output.register_hook(grad_hook))
            return None

        return hook

    def _make_lm_head_hook(self, cap: EmbeddingCapture):
        expected_positions = int(cap.input_ids.size)

        def hook(module, args, output):
            cap.lm_head_forward_count += 1
            if cap.lm_head_forward_count > 1:
                raise ObserverError(
                    f"step {cap.step}: lm_head forward hook fired "
                    f"{cap.lm_head_forward_count} times; exactly one forward "
                    "per captured step is required"
                )
            if not args or not isinstance(args[0], torch.Tensor):
                raise ObserverError(
                    f"step {cap.step}: lm_head did not receive a tensor input"
                )
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise GeometryMismatchError(
                    f"step {cap.step}: lm_head input shape "
                    f"{tuple(hidden.shape)}, expected (1, T, D)"
                )
            if int(hidden.shape[1]) != expected_positions:
                raise GeometryMismatchError(
                    f"step {cap.step}: lm_head input has {int(hidden.shape[1])} "
                    f"positions but input_ids has {expected_positions}"
                )
            if int(hidden.shape[2]) != self.table.hidden_size:
                raise GeometryMismatchError(
                    f"step {cap.step}: lm_head input width "
                    f"{int(hidden.shape[2])} != hidden size "
                    f"{self.table.hidden_size}"
                )
            # Live vocabulary-axis agreement against the ACTUAL logits.
            if isinstance(output, torch.Tensor):
                if int(output.shape[-1]) != self.table.vocab_size:
                    raise GeometryMismatchError(
                        f"step {cap.step}: logits vocabulary dimension "
                        f"{int(output.shape[-1])} != tied table rows "
                        f"{self.table.vocab_size}"
                    )
            cap.lm_head_input = hidden.detach().clone()
            return None

        return hook

    @contextmanager
    def capture(self, step: int, input_ids, targets):
        """Spans forward -> backward -> clip -> step -> post-step snapshot.

        Hooks exist only inside this context and only on captured steps. The
        ``finally`` removes them and releases snapshots even if forward, loss,
        backward, clip, step, derivation, or writing raises.
        """
        if not self.should_capture(step):
            raise ObserverError(
                f"capture requested for step {step}, outside window "
                f"{sorted(self.capture_steps)}"
            )
        ids_arr = np.asarray(
            input_ids.detach().cpu().numpy() if torch.is_tensor(input_ids)
            else input_ids
        ).reshape(-1).astype(np.int64)
        tgt_arr = np.asarray(
            targets.detach().cpu().numpy() if torch.is_tensor(targets)
            else targets
        ).reshape(-1).astype(np.int64)
        if ids_arr.size != tgt_arr.size:
            raise ObserverError(
                f"step {step}: input_ids has {ids_arr.size} positions but "
                f"targets has {tgt_arr.size}; the control's causal contract is "
                "input_ids = ids[:-1], targets = ids[1:]"
            )
        V = self.table.vocab_size
        for name, arr in (("input_ids", ids_arr), ("targets", tgt_arr)):
            if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= V):
                raise ObserverError(
                    f"step {step}: {name} contains ids outside [0, {V})"
                )

        cap = EmbeddingCapture(step=int(step), input_ids=ids_arr, targets=tgt_arr)
        try:
            self._handles.append(
                self.table.embedding_module.register_forward_hook(
                    self._make_embedding_hook(cap)
                )
            )
            self._handles.append(
                self.table.lm_head_module.register_forward_hook(
                    self._make_lm_head_hook(cap)
                )
            )
            yield cap
        finally:
            self._remove_hooks()
            cap.release_snapshots()

    def _require_stage(self, cap: EmbeddingCapture, stage: str) -> None:
        expected = (
            STAGES[len(cap.stages_done)]
            if len(cap.stages_done) < len(STAGES) else None
        )
        if stage != expected:
            raise ObserverError(
                f"step {cap.step}: stage {stage!r} recorded out of order; "
                f"stages so far {cap.stages_done}, expected {expected!r}"
            )
        cap.stages_done.append(stage)

    # -- stage 1: after backward, before clip ---------------------------

    def record_pre_clip(self, cap: EmbeddingCapture) -> None:
        """Native table clone, step-20 baseline, and pre-clip row statistics."""
        self._require_stage(cap, "pre_clip")

        if cap.embedding_forward_count != 1:
            raise ObserverError(
                f"step {cap.step}: embedding forward hook fired "
                f"{cap.embedding_forward_count} times, expected exactly 1"
            )
        if cap.lm_head_forward_count != 1:
            raise ObserverError(
                f"step {cap.step}: lm_head forward hook fired "
                f"{cap.lm_head_forward_count} times, expected exactly 1"
            )
        if cap.embedding_grad_count != 1 or cap.embedding_output_grad is None:
            raise ObserverError(
                f"step {cap.step}: embedding gradient hook fired "
                f"{cap.embedding_grad_count} times; backward missing or "
                "incomplete"
            )
        if cap.lm_head_input is None:
            raise ObserverError(f"step {cap.step}: lm_head input never captured")

        # Module hooks have done their work; the tensor hook already fired.
        self._remove_hooks()

        param = self.table.param
        grad = param.grad
        if grad is None:
            raise ObserverError(
                f"step {cap.step}: tied table has no gradient; AdamW would "
                "skip it entirely and there would be nothing to observe"
            )

        cap.table_pre = param.detach().clone()
        cap.grad_pre_clip = grad.detach().clone()

        if cap.step == CAPTURE_START_STEP:
            # The one-time full baseline that anchors the whole state chain.
            cap.baseline_bits = bf16_to_bits(cap.table_pre)

        V, D = self.table.vocab_size, self.table.hidden_size

        pre32 = cap.table_pre.to(torch.float32)
        grad32 = cap.grad_pre_clip.to(torch.float32)
        self._assert_finite(cap, "tied table", pre32)
        self._assert_finite(cap, "pre-clip tied gradient", grad32)

        cap.row_metrics["pre_value_l2"] = self._row_l2(pre32)
        cap.row_metrics["pre_value_rms"] = (
            cap.row_metrics["pre_value_l2"] / np.sqrt(float(D))
        )
        cap.row_metrics["pre_clip_grad_l2"] = self._row_l2(grad32)
        cap.row_metrics["pre_clip_grad_max_abs"] = self._row_max_abs(grad32)

        # Reconstruct the lookup-path gradient by scatter-add of the ACTUAL
        # embedding-output gradient over the ACTUAL input ids. Repeated tokens
        # accumulate, which is the whole point of the scatter-add.
        emb_grad32 = cap.embedding_output_grad.reshape(-1, D).to(torch.float32)
        self._assert_finite(cap, "embedding output gradient", emb_grad32)
        unique_rows, inverse = np.unique(cap.input_ids, return_inverse=True)
        input_path = torch.zeros(
            (unique_rows.size, D), dtype=torch.float32, device=emb_grad32.device
        )
        input_path.index_add_(
            0, torch.as_tensor(inverse, dtype=torch.long,
                               device=emb_grad32.device), emb_grad32
        )
        cap.detail["input_path_rows"] = unique_rows.astype(np.int64)
        cap.detail["input_path_grad"] = input_path

        input_path_l2 = np.zeros((V,), dtype=np.float32)
        input_path_l2[unique_rows] = (
            torch.linalg.vector_norm(input_path, dim=1).cpu().numpy()
        )
        cap.row_metrics["input_path_grad_l2"] = input_path_l2

        # Algebraic residual: total minus reconstructed lookup path. For rows
        # absent from input_ids the reconstruction is exactly zero, so the
        # residual equals the total gradient there.
        residual_l2 = cap.row_metrics["pre_clip_grad_l2"].copy()
        if unique_rows.size:
            rows_t = torch.as_tensor(
                unique_rows, dtype=torch.long, device=grad32.device
            )
            residual_rows = grad32.index_select(0, rows_t) - input_path
            residual_l2[unique_rows] = (
                torch.linalg.vector_norm(residual_rows, dim=1).cpu().numpy()
            )
        cap.row_metrics["output_residual_grad_l2"] = residual_l2

        # Token roles from the actual trainer tensors.
        input_count = np.bincount(cap.input_ids, minlength=V).astype(np.int64)
        target_count = np.bincount(cap.targets, minlength=V).astype(np.int64)
        role = np.where(
            (input_count > 0) & (target_count > 0), ROLE_BOTH,
            np.where(input_count > 0, ROLE_INPUT_ONLY,
                     np.where(target_count > 0, ROLE_TARGET_ONLY, ROLE_NEITHER)),
        ).astype(np.int64)
        cap.row_counts["input_count"] = input_count
        cap.row_counts["target_count"] = target_count
        cap.row_counts["role_code"] = role

    # -- stage 2: after the trainer's clip_grad_norm_, before step ------

    def record_clip(self, cap: EmbeddingCapture, returned_norm) -> None:
        """Returned global norm plus post-clip row statistics from live .grad."""
        self._require_stage(cap, "clip")

        value = float(returned_norm)
        if not np.isfinite(value):
            raise NonFiniteCaptureError(
                f"step {cap.step}: clip_grad_norm_ returned {value}"
            )
        cap.returned_grad_norm = value
        cap.learning_rate = float(
            self.optimizer.param_groups[self.table.group_index]["lr"]
        )

        grad = self.table.param.grad
        if grad is None:
            raise ObserverError(
                f"step {cap.step}: tied table gradient disappeared before the "
                "optimizer step"
            )
        # Retained through record_post_step: the detail-row union is not known
        # until the delta top-k has been computed.
        cap.grad_post_clip = grad.detach().clone()
        grad32 = cap.grad_post_clip.to(torch.float32)
        self._assert_finite(cap, "post-clip tied gradient", grad32)
        cap.row_metrics["post_clip_grad_l2"] = self._row_l2(grad32)
        cap.row_metrics["post_clip_grad_max_abs"] = self._row_max_abs(grad32)

    # -- stage 3: after the trainer's optimizer.step() ------------------

    def record_post_step(self, cap: EmbeddingCapture) -> None:
        """Exact native patch, row movement, alignments, and detail rows."""
        self._require_stage(cap, "post_step")

        if cap.table_pre is None:
            raise ObserverError(
                f"step {cap.step}: no pre-step table snapshot; an exact native "
                "patch cannot be built"
            )
        V, D = self.table.vocab_size, self.table.hidden_size

        # Patch membership is integer bit inequality over the FULL table. The
        # sparse patch only proves "changed coordinates" because this
        # comparison is exhaustive, never sampled and never thresholded.
        pre_bits = bf16_to_bits(cap.table_pre).reshape(-1)
        post_bits = bf16_to_bits(self.table.param.detach()).reshape(-1)
        if pre_bits.shape != post_bits.shape:
            raise GeometryMismatchError(
                f"step {cap.step}: table shape changed during the update"
            )
        changed_mask = pre_bits != post_bits
        changed_index = np.flatnonzero(changed_mask).astype(np.int64)
        changed_pre = pre_bits[changed_index].copy()
        changed_post = post_bits[changed_index].copy()
        if np.any(changed_pre == changed_post):
            raise ObserverError(
                f"step {cap.step}: a patch member has equal pre and post bits"
            )
        if changed_index.size and not np.all(np.diff(changed_index) > 0):
            raise ObserverError(
                f"step {cap.step}: patch indices are not strictly increasing"
            )

        cap.patch = {
            "changed_flat_index": changed_index,
            "changed_pre_bf16_bits": changed_pre,
            "changed_post_bf16_bits": changed_post,
        }

        changed_rows = (
            changed_index // D if changed_index.size
            else np.zeros((0,), dtype=np.int64)
        )
        changed_count = np.bincount(changed_rows, minlength=V).astype(np.int64)
        cap.row_counts["changed_coordinate_count"] = changed_count
        cap.row_counts["changed_flag"] = (changed_count > 0).astype(np.int64)

        # Float32 deltas come ONLY from the stored bits, by exact expansion.
        delta_values = (
            bits_to_float32(changed_post) - bits_to_float32(changed_pre)
            if changed_index.size
            else np.zeros((0,), dtype=np.float32)
        )
        if changed_index.size and not np.all(np.isfinite(delta_values)):
            raise NonFiniteCaptureError(
                f"step {cap.step}: non-finite value in the native delta"
            )

        delta_sq = np.zeros((V,), dtype=np.float64)
        delta_max = np.zeros((V,), dtype=np.float64)
        if changed_index.size:
            np.add.at(delta_sq, changed_rows, delta_values.astype(np.float64) ** 2)
            np.maximum.at(delta_max, changed_rows, np.abs(delta_values))
        cap.row_metrics["delta_l2"] = np.sqrt(delta_sq).astype(np.float32)
        cap.row_metrics["delta_max_abs"] = delta_max.astype(np.float32)

        # Alignments. Delta is exactly zero off the patch, so every inner
        # product is a sparse gather over changed coordinates only.
        grad_pre32 = cap.grad_pre_clip.to(torch.float32).reshape(-1).cpu().numpy()
        cap.row_metrics.update(self._alignment(
            V, changed_rows, changed_index, delta_values,
            grad_pre32, cap.row_metrics["pre_clip_grad_l2"],
            "delta_grad",
        ))

        residual_full = grad_pre32.copy()
        input_rows = cap.detail["input_path_rows"]
        if input_rows.size:
            ip = cap.detail["input_path_grad"].cpu().numpy().reshape(-1)
            flat_slots = (
                input_rows[:, None] * D + np.arange(D, dtype=np.int64)[None, :]
            ).reshape(-1)
            residual_full[flat_slots] -= ip
        cap.row_metrics.update(self._alignment(
            V, changed_rows, changed_index, delta_values,
            residual_full, cap.row_metrics["output_residual_grad_l2"],
            "delta_residual",
        ))

        input_path_full = np.zeros_like(grad_pre32)
        if input_rows.size:
            input_path_full[flat_slots] = ip
        cap.row_metrics.update(self._alignment(
            V, changed_rows, changed_index, delta_values,
            input_path_full, cap.row_metrics["input_path_grad_l2"],
            "delta_input_grad",
        ))

        # Relative movement, masked where the pre-update row norm is zero.
        pre_l2 = cap.row_metrics["pre_value_l2"]
        rel = np.full((V,), np.nan, dtype=np.float32)
        rel_valid = pre_l2 > 0
        rel[rel_valid] = (
            cap.row_metrics["delta_l2"][rel_valid] / pre_l2[rel_valid]
        )
        cap.row_metrics["relative_delta_l2"] = rel
        cap.row_metrics["relative_delta_l2_valid"] = rel_valid.astype(np.uint8)

        self._select_detail_rows(cap)

        # Model-sized temporaries die here, every step.
        cap.release_snapshots()

    def _alignment(self, V, changed_rows, changed_index, delta_values,
                   other_flat, other_row_l2, prefix) -> dict:
        """Row-wise inner product and cosine against a comparison field.

        Cosine is valid ONLY when both norms are nonzero; invalid entries are
        NaN with a paired validity mask. Zero would mean orthogonal, which is
        an assertion the measurement does not support.
        """
        inner = np.zeros((V,), dtype=np.float64)
        if changed_index.size:
            np.add.at(
                inner, changed_rows,
                delta_values.astype(np.float64) * other_flat[changed_index],
            )
        delta_l2 = np.zeros((V,), dtype=np.float64)
        np.add.at(delta_l2, changed_rows, delta_values.astype(np.float64) ** 2)
        delta_l2 = np.sqrt(delta_l2)
        other_l2 = np.asarray(other_row_l2, dtype=np.float64)
        valid = (delta_l2 > 0) & (other_l2 > 0)
        cosine = np.full((V,), np.nan, dtype=np.float32)
        cosine[valid] = (
            inner[valid] / (delta_l2[valid] * other_l2[valid])
        ).astype(np.float32)
        return {
            f"{prefix}_inner": inner.astype(np.float32),
            f"{prefix}_cosine": cosine,
            f"{prefix}_cosine_valid": valid.astype(np.uint8),
        }

    def _select_detail_rows(self, cap: EmbeddingCapture) -> None:
        """Deterministic union: current token rows + top-256 delta + top-256 grad.

        Selection affects ONLY extra gradient detail. It never affects the exact
        all-coordinate patch or the all-row scalar metrics.
        """
        present = np.union1d(cap.input_ids, cap.targets).astype(np.int64)
        top_delta = rank_descending_stable(
            cap.row_metrics["delta_l2"], DETAIL_TOP_K_DELTA
        )
        top_grad = rank_descending_stable(
            cap.row_metrics["pre_clip_grad_l2"], DETAIL_TOP_K_GRAD
        )
        rows = np.unique(np.concatenate([present, top_delta, top_grad]))
        cap.detail["detail_row_ids"] = rows.astype(np.int64)

        rows_t = torch.as_tensor(
            rows, dtype=torch.long, device=cap.grad_pre_clip.device
        )
        cap.detail["detail_pre_clip_grad_bits"] = bf16_to_bits(
            cap.grad_pre_clip.index_select(0, rows_t)
        )
        cap.detail["detail_post_clip_grad_bits"] = bf16_to_bits(
            cap.grad_post_clip.index_select(0, rows_t)
        )

    # ------------------------------------------------------------------
    # completeness + derivation
    # ------------------------------------------------------------------

    def _assert_finite(self, cap: EmbeddingCapture, label: str,
                       tensor: torch.Tensor) -> None:
        if not bool(torch.isfinite(tensor).all()):
            raise NonFiniteCaptureError(
                f"step {cap.step}: non-finite value in {label}"
            )

    @staticmethod
    def _row_l2(matrix: torch.Tensor) -> np.ndarray:
        return torch.linalg.vector_norm(
            matrix, dim=1
        ).to(torch.float32).cpu().numpy()

    @staticmethod
    def _row_max_abs(matrix: torch.Tensor) -> np.ndarray:
        return matrix.abs().amax(dim=1).to(torch.float32).cpu().numpy()

    def assert_complete(self, cap: EmbeddingCapture) -> None:
        if cap.stages_done != list(STAGES):
            raise ObserverError(
                f"step {cap.step}: stages recorded {cap.stages_done} != "
                f"required {list(STAGES)}"
            )
        V = self.table.vocab_size
        for name in ROW_METRIC_FIELDS:
            arr = cap.row_metrics.get(name)
            if arr is None or arr.shape != (V,):
                raise ObserverError(
                    f"step {cap.step}: row metric {name} missing or not [V]"
                )
            if name not in ROW_METRIC_MAY_BE_NAN and not np.all(np.isfinite(arr)):
                raise NonFiniteCaptureError(
                    f"step {cap.step}: non-finite value in row metric {name}"
                )
        for name in ROW_VALIDITY_FIELDS:
            if cap.row_metrics.get(name) is None:
                raise ObserverError(
                    f"step {cap.step}: validity mask {name} missing"
                )
        for name in ROW_COUNT_FIELDS:
            if cap.row_counts.get(name) is None:
                raise ObserverError(f"step {cap.step}: row count {name} missing")
        if not cap.patch:
            raise ObserverError(f"step {cap.step}: native patch never built")
        if (cap.step == CAPTURE_START_STEP) != (cap.baseline_bits is not None):
            raise ObserverError(
                f"step {cap.step}: the full pre-window baseline belongs to step "
                f"{CAPTURE_START_STEP} alone"
            )
        if self.hooks_active():
            raise ObserverError(f"step {cap.step}: hook handles survived")
        if (cap.table_pre is not None or cap.grad_pre_clip is not None
                or cap.grad_post_clip is not None):
            raise ObserverError(
                f"step {cap.step}: temporary snapshots survived the step"
            )

    def derive(self, cap: EmbeddingCapture, token_ids, *, training_loss: float,
               dataset_row: int, optimizer_step: int) -> tto.DerivedCapture:
        """Package the staged capture into a DerivedCapture."""
        self.assert_complete(cap)
        if not np.isfinite(training_loss):
            raise NonFiniteCaptureError(
                f"step {cap.step}: non-finite training loss {training_loss}"
            )

        arrays: dict = {}
        records: list = []
        raw_bytes = 0

        def _record(key, name, arr, source_dtype, storage_dtype, numeric=True):
            nonlocal raw_bytes
            arr = np.asarray(arr)
            # ascontiguousarray promotes 0-d to shape (1,); scalars in this
            # schema must stay 0-d so `int(...)` round-trips.
            if arr.ndim > 0:
                arr = np.ascontiguousarray(arr)
            arrays[key] = arr
            raw_bytes += arr.nbytes
            total = int(arr.size)
            finite = (
                int(np.count_nonzero(np.isfinite(arr)))
                if numeric and np.issubdtype(arr.dtype, np.floating) else total
            )
            records.append({
                "key": key,
                "layer": None,
                "name": name,
                "shape": list(arr.shape),
                "source_dtype": source_dtype,
                "storage_dtype": storage_dtype,
                "finite_count": finite,
                "nonfinite_count": total - finite,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "raw_bytes": int(arr.nbytes),
            })

        V, D = self.table.vocab_size, self.table.hidden_size

        # -- exact native state chain
        if cap.baseline_bits is not None:
            _record("pre_window_bf16_bits", "pre_window_bf16_bits",
                    cap.baseline_bits.reshape(V, D), "bfloat16_bits", "uint16")
        for key in ("changed_flat_index", "changed_pre_bf16_bits",
                    "changed_post_bf16_bits"):
            _record(key, key, cap.patch[key], "bfloat16_bits",
                    str(cap.patch[key].dtype))
        _record("table_shape", "table_shape",
                np.asarray([V, D], dtype="<i8"), "int64", "int64")
        _record("changed_coordinate_total", "changed_coordinate_total",
                np.asarray(int(cap.patch["changed_flat_index"].size),
                           dtype="<i8"), "int64", "int64")
        _record("changed_row_total", "changed_row_total",
                np.asarray(int(np.count_nonzero(
                    cap.row_counts["changed_flag"])), dtype="<i8"),
                "int64", "int64")

        # -- exact boundary detail
        _record("embedding_output_grad_bf16_bits",
                "embedding_output_grad_bf16_bits",
                bf16_to_bits(cap.embedding_output_grad.reshape(-1, D)),
                "bfloat16_bits", "uint16")
        _record("lm_head_input_bf16_bits", "lm_head_input_bf16_bits",
                bf16_to_bits(cap.lm_head_input.reshape(-1, D)),
                "bfloat16_bits", "uint16")
        _record("detail_row_ids", "detail_row_ids",
                cap.detail["detail_row_ids"], "int64", "int64")
        _record("detail_pre_clip_grad_bf16_bits",
                "detail_pre_clip_grad_bf16_bits",
                cap.detail["detail_pre_clip_grad_bits"], "bfloat16_bits",
                "uint16")
        _record("detail_post_clip_grad_bf16_bits",
                "detail_post_clip_grad_bf16_bits",
                cap.detail["detail_post_clip_grad_bits"], "bfloat16_bits",
                "uint16")

        # -- all-vocabulary row metrics
        for name in ROW_METRIC_FIELDS:
            _record(f"row/{name}", name,
                    np.asarray(cap.row_metrics[name], dtype=np.float32),
                    "float32", "float32")
        for name in ROW_VALIDITY_FIELDS:
            _record(f"row/{name}", name,
                    np.asarray(cap.row_metrics[name], dtype=np.uint8),
                    "bool", "uint8")
        for name in ROW_COUNT_FIELDS:
            _record(f"row/{name}", name,
                    np.asarray(cap.row_counts[name], dtype="<i8"),
                    "int64", "int64")

        # -- identity and token arrays
        ids = [int(i) for i in token_ids]
        supervised_token_count = len(ids) - 1
        if supervised_token_count != int(cap.input_ids.size):
            raise ObserverError(
                f"step {cap.step}: supervised_token_count "
                f"{supervised_token_count} != captured positions "
                f"{int(cap.input_ids.size)}"
            )
        _record("token_ids", "token_ids", np.asarray(ids, dtype="<i8"),
                "int64", "int64")
        _record("input_ids", "input_ids", cap.input_ids, "int64", "int64")
        _record("targets", "targets", cap.targets, "int64", "int64")
        _record("training_loss", "training_loss",
                np.float32(training_loss).reshape(()), "float32", "float32")
        _record("returned_grad_norm", "returned_grad_norm",
                np.float32(cap.returned_grad_norm).reshape(()),
                "float32", "float32")
        _record("learning_rate", "learning_rate",
                np.float32(cap.learning_rate).reshape(()), "float32", "float32")
        _record("supervised_token_count", "supervised_token_count",
                np.asarray(supervised_token_count, dtype="<i8"),
                "int64", "int64")
        _record("dataset_row", "dataset_row",
                np.asarray(int(dataset_row), dtype="<i8"), "int64", "int64")
        _record("optimizer_step", "optimizer_step",
                np.asarray(int(optimizer_step), dtype="<i8"), "int64", "int64")
        _record("input_ids_sha256", "input_ids_sha256",
                np.asarray(token_ids_sha256(cap.input_ids.tolist()),
                           dtype=np.str_).reshape(()), "str", "str",
                numeric=False)
        _record("targets_sha256", "targets_sha256",
                np.asarray(token_ids_sha256(cap.targets.tolist()),
                           dtype=np.str_).reshape(()), "str", "str",
                numeric=False)

        return tto.DerivedCapture(
            step=cap.step,
            phase=PHASE,
            arrays=arrays,
            tensor_records=records,
            token_ids=ids,
            token_sha256=token_ids_sha256(ids),
            sequence_length=supervised_token_count,
            raw_bytes=raw_bytes,
        )


# ---------------------------------------------------------------------------
# compact W&B scalar summaries (no token text, no semantic verdict)
# ---------------------------------------------------------------------------


def observer_scalars(derived: tto.DerivedCapture) -> dict:
    """Descriptive scalars only, under observer/embedding_learning/*."""
    a = derived.arrays
    V, D = (int(x) for x in a["table_shape"])
    changed_flag = a["row/changed_flag"]
    delta_l2 = a["row/delta_l2"]
    input_count = a["row/input_count"]
    target_count = a["row/target_count"]
    changed_total = int(a["changed_coordinate_total"])
    changed_rows = int(a["changed_row_total"])

    input_rows = input_count > 0
    target_rows = target_count > 0
    moved = changed_flag > 0
    moved_delta = delta_l2[moved]
    valid = a["row/delta_grad_cosine_valid"]
    detail_ids = a["detail_row_ids"]

    def _frac(mask):
        total = int(np.count_nonzero(mask))
        return (
            float(np.count_nonzero(mask & moved) / total) if total else 0.0
        )

    return {
        "observer/embedding_learning/changed_coordinate_count": changed_total,
        "observer/embedding_learning/changed_coordinate_fraction": (
            float(changed_total / (V * D)) if V * D else 0.0
        ),
        "observer/embedding_learning/changed_row_count": changed_rows,
        "observer/embedding_learning/changed_row_fraction": (
            float(changed_rows / V) if V else 0.0
        ),
        "observer/embedding_learning/input_present_row_count": int(
            np.count_nonzero(input_rows)
        ),
        "observer/embedding_learning/input_present_row_fraction_changed":
            _frac(input_rows),
        "observer/embedding_learning/target_present_row_count": int(
            np.count_nonzero(target_rows)
        ),
        "observer/embedding_learning/target_present_row_fraction_changed":
            _frac(target_rows),
        "observer/embedding_learning/changed_row_delta_l2_mean": (
            float(np.mean(moved_delta)) if moved_delta.size else 0.0
        ),
        "observer/embedding_learning/changed_row_delta_l2_median": (
            float(np.median(moved_delta)) if moved_delta.size else 0.0
        ),
        "observer/embedding_learning/detail_row_count": int(detail_ids.size),
        "observer/embedding_learning/detail_row_cosine_valid_fraction": (
            float(np.count_nonzero(valid[detail_ids]) / detail_ids.size)
            if detail_ids.size else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# artifact writing -- same acyclic finalization order as the siblings
# ---------------------------------------------------------------------------


def expected_filenames() -> list:
    return [
        f"step_{step:06d}_{PHASE}.npz"
        for step in range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1)
    ]


def finalize_embedding_learning_dir(out_dir, file_records: list, run_meta: dict,
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
        "baseline_note": (
            "pre_window_bf16_bits is a one-time full-table quantity owned by "
            f"step {CAPTURE_START_STEP} alone; steps "
            f"{CAPTURE_START_STEP + 1}-{CAPTURE_END_STEP} carry only sparse "
            "patches"
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
        "embedding_learning_schema_version": EMBEDDING_LEARNING_SCHEMA_VERSION,
        "camera_number": CAMERA_NUMBER,
        "camera_family": CAMERA_FAMILY,
        "observation_stage": OBSERVATION_STAGE,
        "state_encoding": STATE_ENCODING,
        "patch_membership": PATCH_MEMBERSHIP,
        "patch_membership_assertion": (
            "membership was constructed from an exhaustive native bit "
            "comparison over the FULL table, not sampled and not thresholded; "
            "a sparse patch proves changed coordinates only because of this"
        ),
        "delta_precision": DELTA_PRECISION,
        "baseline_step": CAPTURE_START_STEP,
        "output_residual_semantics": RESIDUAL_SEMANTICS,
        "output_residual_caveat": (
            "output_residual_grad = total_tied_parameter_grad - "
            "reconstructed_input_grad is an ALGEBRAIC residual; it is NOT an "
            "independently captured LM-head branch gradient, and native "
            "accumulation may differ from the float32 reconstruction at "
            "rounding scale"
        ),
        "frozen_hidden_displacement_semantics": DISPLACEMENT_SEMANTICS,
        "frozen_hidden_displacement_caveat": (
            "h_pre @ delta_row holds the pre-update hidden state FIXED and "
            "isolates the tied table's local contribution; it is NOT the full "
            "post-update logit change and is not causal attribution"
        ),
        "tied_table": run_meta.get("tied_table"),
        "role_codes": {str(k): v for k, v in ROLE_NAMES.items()},
        "row_metric_fields": list(ROW_METRIC_FIELDS),
        "row_metric_may_be_nan": sorted(ROW_METRIC_MAY_BE_NAN),
        "row_validity_fields": list(ROW_VALIDITY_FIELDS),
        "row_count_fields": list(ROW_COUNT_FIELDS),
        "detail_top_k_delta": DETAIL_TOP_K_DELTA,
        "detail_top_k_grad": DETAIL_TOP_K_GRAD,
        "detail_selection_semantics": (
            "union of current input/target token rows, top-256 realized "
            "delta_l2 and top-256 pre-clip total-gradient l2, ties broken by "
            "ascending token id; selection affects ONLY extra gradient detail "
            "and never the exact patch or the all-row metrics"
        ),
        "duplicates_camera3": False,
        "semantic_claims": "none",
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

    hashes["manifest.json"] = sha256_file(manifest_path)
    lines = [f"{hashes[name]}  {name}" for name in sorted(hashes)]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    return manifest
