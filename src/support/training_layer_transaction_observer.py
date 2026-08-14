"""Layer Transaction Camera for Mamba-3 (steps 20-25, pre/post update).

Modal-free by design so every mechanism is CPU-testable. The Modal trainer
(modal_layer_transaction_observer.py) imports this module; nothing here
imports modal, wandb, or CUDA-only code.

This camera builds the shared layer-token coordinate system for the later
observer cameras. During optimizer updates 20-25 of the frozen 50-update
GSM8K trajectory it captures, for every loaded block:

    * the REAL pre-update training forward under theta_(k-1): the live
      tensors crossing each block boundary (block_input, mixer_input,
      mixer_output, mlp_input/mlp_output when the block has an MLP, and
      block_residual -- output[1] of Block.forward, the residual stream
      the block threads to the next block via its return value) plus the
      final norm output;
    * the ordinary loss gradient delivered to those same boundary tensors
      during the ONE real loss.backward() (pre-update only);
    * paired post-update values from a replay of the identical token IDs
      under theta_k with torch.no_grad().

Gradients are raw autograd values of the MEAN next-token cross-entropy,
observed pre-clipping. They are NOT parameter gradients, NOT clipped
gradients, NOT applied updates, NOT the Jacobian. A derived
descent_pressure = -gradient view is an analyzer-side reporting label only;
stored gradients keep their original sign.

What this camera does NOT observe: recurrent cache state or any
source-derived recurrent state, B/C recurrence tensors, embeddings weights,
full logits arrays, Jacobians, optimizer moments, parameter snapshots,
canaries, causal interventions or patching.

Infrastructure reuse: RNG discipline, deterministic NPZ writing, token
hashing, SHA-256 helpers and the error classes come from
``training_time_observer`` (Camera 1, untouched). Block anatomy (Block with
``norm`` + ``mixer``, optional ``mlp``, ``fused_add_norm`` routing, and the
``(hidden_states, residual)`` tuple return) is discovered at runtime from
the loaded model -- nothing about the deployed contract is inferred from
the prototype norm-free block, and no boundary is ever recomputed: every
captured tensor is a live tensor that genuinely crosses the boundary
(Lesson 1). In particular:

    * ``fused_add_norm`` is read on the loaded model at construction time,
      not inferred: when the backbone takes the fused path, ``norm_f`` is
      never invoked as a module and the genuine final-norm output tensor is
      the lm_head input, so the capture site switches to an lm_head
      pre-hook (same live tensor; captured, never recomputed);
    * Block.forward returns ``(hidden_states, residual)``. The residual is
      threaded through the return value, so ``block_output`` = output[0]
      is NOT the residual stream (and on MLP blocks it is byte-identical to
      mlp_output). This camera captures ``block_residual`` = output[1]
      instead -- an ordinary exposed layer tensor (Lesson 13-compliant),
      float32 under the deployed ``residual_in_fp32`` config.
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

# Canonical module identities (mamba3_core registration) and shared helpers
# are established by this import; this camera reuses Camera 1's infrastructure
# verbatim. No second implementation of RNG discipline or the NPZ writer.
import training_time_observer as tto  # noqa: E402

ObserverError = tto.ObserverError
NonFiniteCaptureError = tto.NonFiniteCaptureError
GeometryMismatchError = tto.GeometryMismatchError

LTO_SCHEMA_VERSION = 1
CAMERA_NUMBER = 6
CAMERA_FAMILY = "layer_transaction"
CAPTURE_START_STEP = 20
CAPTURE_END_STEP = 25
CAPTURE_WINDOW = frozenset(range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1))
PHASES = ("pre_update", "post_update")

# Boundary names as stored. Per-block value boundaries. NOTE: there is no
# "block_output" boundary -- Block.forward returns (hidden_states, residual);
# output[0] is the sublayer output (byte-identical to mlp_output on MLP
# blocks) and output[1] is the residual stream this camera actually wants.
BASE_BOUNDARIES = ("block_input", "mixer_input", "mixer_output")
MLP_BOUNDARIES = ("mlp_input", "mlp_output")
RESIDUAL_BOUNDARY = "block_residual"  # Block.forward output[1]
GLOBAL_BOUNDARY = "final_norm_output"
GRAD_PREFIX = "grad_"

GLOBAL_NAMES = ("token_ids", "supervised_token_count", "causal_target_offset",
                "dataset_row", "optimizer_step")
# pre_update files additionally carry "training_loss";
# post_update files additionally carry "post_update_loss".


# ---------------------------------------------------------------------------
# runtime block discovery
# ---------------------------------------------------------------------------


def discover_blocks(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Find every Block module of the loaded model, in registration order.

    The Block contract is the upstream ``mamba_ssm.modules.block.Block``:
    attributes ``norm`` and ``mixer`` (both nn.Modules), optional attribute
    ``mlp`` (an nn.Module when the block has an MLP, else None/absent).
    Nothing is assumed about the count; discovered, never hardcoded."""
    found = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(getattr(mod, "norm", None), torch.nn.Module)
        and isinstance(getattr(mod, "mixer", None), torch.nn.Module)
    ]
    if not found:
        raise ObserverError(
            "no blocks found: no module exposes both `norm` and `mixer` as "
            "nn.Modules; the loaded model does not match the Block contract"
        )
    return found


def discover_final_norm(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    """Find the backbone final norm (``norm_f``). Exactly one must exist."""
    found = [
        (name, mod)
        for name, mod in model.named_modules()
        if name.rsplit(".", 1)[-1] == "norm_f"
        and isinstance(mod, torch.nn.Module)
    ]
    if len(found) != 1:
        raise ObserverError(
            f"expected exactly one final norm module named `norm_f`, found "
            f"{len(found)}: {[n for n, _ in found]}"
        )
    return found[0]


def discover_lm_head(model: torch.nn.Module) -> tuple[str, torch.nn.Module] | None:
    """Find the LM head module (``lm_head``), or None when the model has no
    module with that name. Exactly one lm_head may exist."""
    found = [
        (name, mod)
        for name, mod in model.named_modules()
        if name.rsplit(".", 1)[-1] == "lm_head"
        and isinstance(mod, torch.nn.Module)
    ]
    if len(found) > 1:
        raise ObserverError(
            f"expected at most one lm_head module, found "
            f"{[n for n, _ in found]}"
        )
    return found[0] if found else None


# ---------------------------------------------------------------------------
# capture containers
# ---------------------------------------------------------------------------


@dataclass
class BoundaryCapture:
    """Raw hook output for one (step, phase). Torch tensors, detached."""

    step: int
    phase: str
    values: dict = field(default_factory=dict)         # layer idx -> {name: t}
    grads: dict = field(default_factory=dict)          # layer idx -> {name: t}
    source_dtypes: dict = field(default_factory=dict)  # layer idx -> {name: str}
    grad_source_dtypes: dict = field(default_factory=dict)
    final_value: torch.Tensor | None = None
    final_value_dtype: str | None = None
    final_grad: torch.Tensor | None = None
    final_grad_dtype: str | None = None


# ---------------------------------------------------------------------------
# observer
# ---------------------------------------------------------------------------


class TrainingLayerTransactionObserver:
    """Temporary pre-hook/forward-hook + tensor-gradient-hook observer.

    All hooks exist ONLY inside an active ``capture`` context. In the
    pre_update phase (the real training forward + backward) a tensor gradient
    hook is registered on each ACTUAL live boundary tensor; the hook stores a
    detached compact clone of the delivered gradient and returns None -- the
    gradient stream is never replaced or altered. In the post_update phase
    (identical tokens under theta_k, torch.no_grad()) only values are
    captured and no tensor hooks are possible or registered.

    Every module and tensor handle is removed in ``finally``, surviving
    forward, loss and backward exceptions. ``retain_grad()`` is never used.
    """

    def __init__(self, model: torch.nn.Module, capture_steps=None):
        # Defaults bind at import; resolve here so the window can be retargeted.
        capture_steps = CAPTURE_WINDOW if capture_steps is None else capture_steps
        self.capture_steps = frozenset(int(s) for s in capture_steps)
        self.blocks = discover_blocks(model)
        self.final_norm_name, self.final_norm = discover_final_norm(model)

        # Lesson 14: the fused routing of the DEPLOYED config is read off the
        # loaded modules at construction time -- never inferred from source.
        # Under fused_add_norm the backbone calls layer_norm_fn(...) and never
        # invokes norm_f.__call__, so the live final-norm output tensor is the
        # lm_head input and the capture site must switch accordingly.
        block_fused = [getattr(blk, "fused_add_norm", None) for _, blk in self.blocks]
        backbone = getattr(model, "backbone", None)
        backbone_fused = getattr(backbone, "fused_add_norm", None)
        known_fused = [bool(f) for f in block_fused + [backbone_fused]
                       if f is not None]
        if not known_fused:
            # No fused flags anywhere: the classic module-call contract
            # (norm_f invoked as a module) -- e.g. the toy test models.
            self.fused_add_norm = False
        else:
            distinct = set(known_fused)
            if len(distinct) != 1:
                raise ObserverError(
                    f"inconsistent fused_add_norm flags across discovered "
                    f"blocks/backbone: block flags {block_fused}, backbone "
                    f"flag {backbone_fused}"
                )
            # Uniform flag: a real fused_add_norm=False build takes the
            # norm_f_output site; the deployed fused build takes lm_head.
            self.fused_add_norm = distinct.pop()
        if self.fused_add_norm:
            found_head = discover_lm_head(model)
            if found_head is None:
                raise ObserverError(
                    "fused_add_norm deployment but no lm_head module found; "
                    "the fused final-norm output is only exposed as the "
                    "lm_head input and there is no module to hook"
                )
            self.lm_head_name, self.lm_head = found_head
            self.final_capture_site = "lm_head_input"
        else:
            self.lm_head_name, self.lm_head = None, None
            self.final_capture_site = "norm_f_output"

        has_mlp = []
        for name, block in self.blocks:
            mlp = getattr(block, "mlp", None)
            if mlp is not None and not isinstance(mlp, torch.nn.Module):
                raise ObserverError(
                    f"block {name}: `mlp` attribute is not None and not an "
                    f"nn.Module ({type(mlp).__name__})"
                )
            has_mlp.append(mlp is not None)
        if len(set(has_mlp)) != 1:
            detail = {
                self.blocks[i][0]: has_mlp[i] for i in range(len(self.blocks))
            }
            raise GeometryMismatchError(
                f"MLP presence differs across discovered blocks: {detail}"
            )
        self.has_mlp = has_mlp[0]
        self.boundaries = tuple(
            BASE_BOUNDARIES
            + (MLP_BOUNDARIES if self.has_mlp else ())
            + (RESIDUAL_BOUNDARY,)
        )
        self.geometry = {
            "num_layers": len(self.blocks),
            "has_mlp": self.has_mlp,
            "fused_add_norm": self.fused_add_norm,
            "boundaries_per_layer": list(self.boundaries),
            "global_boundaries": [GLOBAL_BOUNDARY],
            "layer_module_paths": [name for name, _ in self.blocks],
            "final_norm_module_path": self.final_norm_name,
            "final_norm_capture_site": self.final_capture_site,
            "final_norm_capture_module_path": (
                self.lm_head_name if self.final_capture_site == "lm_head_input"
                else self.final_norm_name
            ),
        }
        # Outstanding tensor-gradient-hook count, for leak assertions.
        self._tensor_hooks_outstanding = 0

    # -- step window ---------------------------------------------------------

    def should_capture(self, step: int) -> bool:
        return int(step) in self.capture_steps

    # -- leak detection --------------------------------------------------------

    def hooks_active(self) -> bool:
        """True if ANY module hook on a hooked module OR any tensor-gradient
        hook this observer registered is still outstanding. The trainer's
        Lesson-2 leak assertion inspects both hook classes under one name."""
        modules = [blk for _, blk in self.blocks] + [
            blk.mixer for _, blk in self.blocks
        ] + [
            blk.mlp for _, blk in self.blocks
            if getattr(blk, "mlp", None) is not None
        ] + [self.final_norm]
        if self.lm_head is not None:
            modules.append(self.lm_head)
        module_side = any(
            len(m._forward_hooks) or len(m._forward_pre_hooks) for m in modules
        )
        return module_side or self._tensor_hooks_outstanding > 0

    # Alias used by trainer-side leak assertions.
    module_hooks_active = hooks_active

    # -- live-tensor recording ------------------------------------------------

    def _admit_live(self, cap: BoundaryCapture, layer, name, tensor,
                    tensor_handles):
        """Store a detached compact clone of one live boundary tensor and, in
        the pre_update phase, register a gradient hook on the ACTUAL tensor."""
        if layer is not None:
            slot = cap.values.setdefault(layer, {})
            if name in slot:
                raise ObserverError(
                    f"layer {layer} boundary {name} observed twice within one "
                    f"capture (step {cap.step}, {cap.phase}); unexpected "
                    "forward re-entry"
                )
            slot[name] = tensor.detach().clone()
            cap.source_dtypes.setdefault(layer, {})[name] = str(tensor.dtype)
        else:
            if cap.final_value is not None:
                raise ObserverError(
                    f"{GLOBAL_BOUNDARY} observed twice within one capture "
                    f"(step {cap.step}, {cap.phase})"
                )
            cap.final_value = tensor.detach().clone()
            cap.final_value_dtype = str(tensor.dtype)

        if cap.phase != "pre_update":
            return None

        if not tensor.requires_grad:
            where = f"layer {layer} boundary {name}" if layer is not None \
                else GLOBAL_BOUNDARY
            raise ObserverError(
                f"step {cap.step}: {where} does not require grad; the "
                "pre_update phase must observe the real training forward, "
                "not a no-grad forward"
            )
        expected_shape = tuple(tensor.shape)

        def grad_hook(grad):
            if layer is not None:
                gslot = cap.grads.setdefault(layer, {})
                if name in gslot:
                    raise ObserverError(
                        f"layer {layer} boundary {name} step {cap.step}: "
                        "gradient hook fired twice; unexpected second "
                        "backward delivery"
                    )
                if tuple(grad.shape) != expected_shape:
                    raise ObserverError(
                        f"layer {layer} boundary {name} step {cap.step}: "
                        f"gradient shape {tuple(grad.shape)} != forward "
                        f"shape {expected_shape}"
                    )
                gslot[name] = grad.detach().clone()
                cap.grad_source_dtypes.setdefault(layer, {})[name] = str(
                    grad.dtype
                )
            else:
                if cap.final_grad is not None:
                    raise ObserverError(
                        f"{GLOBAL_BOUNDARY} step {cap.step}: gradient hook "
                        "fired twice; unexpected second backward delivery"
                    )
                cap.final_grad = grad.detach().clone()
                cap.final_grad_dtype = str(grad.dtype)
            # Mandatory: never replace or alter the gradient.
            return None

        tensor_handles.append(tensor.register_hook(grad_hook))
        self._tensor_hooks_outstanding += 1
        return None

    # -- hook factories ---------------------------------------------------------

    def _pre_hook(self, cap, name, tensor_handles):
        layer_idx = name[0]
        boundary = name[1]

        def hook(module, args):
            tensor = args[0]
            if not isinstance(tensor, torch.Tensor):
                raise ObserverError(
                    f"layer {layer_idx} boundary {boundary}: pre-hook input "
                    f"is not a tensor ({type(tensor).__name__})"
                )
            self._admit_live(cap, layer_idx, boundary, tensor, tensor_handles)
            return None

        return hook

    def _output_hook(self, cap, layer_idx, boundary, tensor_handles, index=0):
        def hook(module, args, output):
            if index is None:
                tensor = output
            else:
                # Block.forward contract: returns (hidden_states, residual).
                # Validate the tuple loudly rather than guessing at indices.
                if not (isinstance(output, (tuple, list))
                        and len(output) > index):
                    raise ObserverError(
                        f"layer {layer_idx} boundary {boundary}: expected a "
                        f"tuple/list module output of length > {index}, got "
                        f"{type(output).__name__}"
                    )
                tensor = output[index]
            if not isinstance(tensor, torch.Tensor):
                raise ObserverError(
                    f"layer {layer_idx} boundary {boundary}: captured output "
                    f"element is not a tensor ({type(tensor).__name__})"
                )
            self._admit_live(cap, layer_idx, boundary, tensor, tensor_handles)
            return None

        return hook

    # -- capture context --------------------------------------------------------

    @contextmanager
    def capture(self, step: int, phase: str):
        """Spans forward -> loss -> (pre phase only) loss.backward().

        The trainer owns the single ordinary backward call; this observer
        never calls backward itself. Hooks are removed in finally."""
        if not self.should_capture(step):
            raise ObserverError(
                f"capture requested for step {step}, outside window "
                f"{sorted(self.capture_steps)}"
            )
        if phase not in PHASES:
            raise ObserverError(f"unknown phase {phase!r}; expected one of {PHASES}")
        cap = BoundaryCapture(step=int(step), phase=phase)
        module_handles = []
        tensor_handles: list = []
        try:
            for idx, (_, block) in enumerate(self.blocks):
                module_handles.append(
                    block.register_forward_pre_hook(
                        self._pre_hook(cap, (idx, "block_input"), tensor_handles)
                    )
                )
                module_handles.append(
                    block.mixer.register_forward_pre_hook(
                        self._pre_hook(cap, (idx, "mixer_input"), tensor_handles)
                    )
                )
                module_handles.append(
                    block.mixer.register_forward_hook(
                        self._output_hook(
                            cap, idx, "mixer_output", tensor_handles, None
                        )
                    )
                )
                if self.has_mlp:
                    module_handles.append(
                        block.mlp.register_forward_pre_hook(
                            self._pre_hook(
                                cap, (idx, "mlp_input"), tensor_handles
                            )
                        )
                    )
                    module_handles.append(
                        block.mlp.register_forward_hook(
                            self._output_hook(
                                cap, idx, "mlp_output", tensor_handles, None
                            )
                        )
                    )
                module_handles.append(
                    block.register_forward_hook(
                        self._output_hook(
                            cap, idx, RESIDUAL_BOUNDARY, tensor_handles, 1
                        )
                    )
                )
            if self.final_capture_site == "lm_head_input":
                # Fused deployment: norm_f is never invoked as a module; the
                # genuine fused final-norm output is the lm_head input tensor
                # (a live, requires_grad tensor; captured, never recomputed).
                module_handles.append(
                    self.lm_head.register_forward_pre_hook(
                        self._pre_hook(cap, (None, GLOBAL_BOUNDARY), tensor_handles)
                    )
                )
            else:
                module_handles.append(
                    self.final_norm.register_forward_hook(
                        self._output_hook(
                            cap, None, GLOBAL_BOUNDARY, tensor_handles, None
                        )
                    )
                )
            yield cap
        finally:
            for h in module_handles:
                h.remove()
            n_tensor = len(tensor_handles)
            for h in tensor_handles:
                h.remove()
            self._tensor_hooks_outstanding -= n_tensor

    # ------------------------------------------------------------------
    # completeness + derivation
    # ------------------------------------------------------------------

    def check_capture_complete(self, cap: BoundaryCapture) -> None:
        """Public completeness gate called by the trainer between backward
        and clipping; also invoked inside derive()."""
        self._check_layer_complete(cap)

    def _check_layer_complete(self, cap: BoundaryCapture) -> None:
        expected_layers = set(range(len(self.blocks)))
        missing = []
        for idx in sorted(expected_layers):
            got = set(cap.values.get(idx, {}))
            if got != set(self.boundaries):
                missing.append(
                    f"layer {idx}: boundaries {sorted(got)} != expected "
                    f"{sorted(self.boundaries)}"
                )
        if missing:
            raise ObserverError(
                f"step {cap.step} {cap.phase}: incomplete boundary capture: "
                + "; ".join(missing)
            )
        if cap.final_value is None:
            raise ObserverError(
                f"step {cap.step} {cap.phase}: {GLOBAL_BOUNDARY} not captured"
            )
        if cap.phase == "pre_update":
            for idx in sorted(expected_layers):
                got = set(cap.grads.get(idx, {}))
                if got != set(self.boundaries):
                    raise ObserverError(
                        f"step {cap.step}: no boundary gradient delivered for "
                        f"layer {idx} (have {sorted(got)}); backward missing "
                        "or incomplete"
                    )
            if cap.final_grad is None:
                raise ObserverError(
                    f"step {cap.step}: no gradient delivered at "
                    f"{GLOBAL_BOUNDARY}"
                )
        else:
            if cap.grads or cap.final_grad is not None:
                raise ObserverError(
                    f"step {cap.step} {cap.phase}: gradients present; the "
                    "post_update replay must not produce gradients"
                )

    def derive(self, cap: BoundaryCapture, token_ids, *, loss: float,
               dataset_row: int, optimizer_step: int) -> tto.DerivedCapture:
        """Package boundary values (+ pre-phase boundary gradients) into a
        DerivedCapture. Raises NonFiniteCaptureError on ANY non-finite
        captured value, gradient, or loss."""
        self._check_layer_complete(cap)

        if not np.isfinite(loss):
            raise NonFiniteCaptureError(
                f"step {cap.step} {cap.phase}: non-finite loss {loss}"
            )

        arrays: dict = {}
        records: list = []
        raw_bytes = 0
        seq_len = None

        def _admit(key, layer, name, tensor, source_dtype):
            nonlocal raw_bytes, seq_len
            t = tensor.detach().to(torch.float32).cpu()
            if t.ndim != 3 or t.shape[0] != 1:
                raise ObserverError(
                    f"step {cap.step} {cap.phase} tensor {name}: expected "
                    f"shape (1, T, D), got {tuple(t.shape)}"
                )
            t = t[0]
            finite = int(torch.isfinite(t).sum())
            total = t.numel()
            if finite != total:
                raise NonFiniteCaptureError(
                    f"step {cap.step} {cap.phase} tensor {name}: "
                    f"{total - finite} non-finite of {total}"
                )
            if seq_len is None:
                seq_len = int(t.shape[0])
            elif int(t.shape[0]) != seq_len:
                raise ObserverError(
                    f"step {cap.step} {cap.phase} tensor {name}: time "
                    f"dimension {int(t.shape[0])} != {seq_len}"
                )
            arr = t.numpy()
            arrays[key] = arr
            raw_bytes += arr.nbytes
            records.append({
                "key": key,
                "layer": layer,
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

        for idx in range(len(self.blocks)):
            for name in self.boundaries:
                _admit(
                    f"layer_{idx:02d}/{name}", idx, name,
                    cap.values[idx][name], cap.source_dtypes[idx][name],
                )
            if cap.phase == "pre_update":
                for name in self.boundaries:
                    _admit(
                        f"layer_{idx:02d}/{GRAD_PREFIX}{name}", idx,
                        f"{GRAD_PREFIX}{name}", cap.grads[idx][name],
                        cap.grad_source_dtypes[idx][name],
                    )

        _admit(GLOBAL_BOUNDARY, None, GLOBAL_BOUNDARY, cap.final_value,
               cap.final_value_dtype)
        if cap.phase == "pre_update":
            _admit(f"{GRAD_PREFIX}{GLOBAL_BOUNDARY}", None,
                   f"{GRAD_PREFIX}{GLOBAL_BOUNDARY}", cap.final_grad,
                   cap.final_grad_dtype)

        ids = [int(i) for i in token_ids]
        supervised_token_count = len(ids) - 1  # targets = ids[1:]
        if seq_len is not None and supervised_token_count != seq_len:
            raise ObserverError(
                f"step {cap.step} {cap.phase}: supervised_token_count "
                f"{supervised_token_count} != captured time dimension {seq_len}"
            )
        # Lesson 7: the alignment rule ships WITH the data, not as manifest
        # prose. Captured tensor row t is produced by input ids[t] and is
        # supervised against target ids[t + causal_target_offset]; the
        # offset is always 1 for next-token cross-entropy.
        causal_target_offset = 1

        def _admit_global(name, arr, source_dtype, storage_dtype):
            nonlocal raw_bytes
            arrays[name] = arr
            raw_bytes += arr.nbytes
            records.append({
                "key": name,
                "layer": None,
                "name": name,
                "shape": list(arr.shape),
                "source_dtype": source_dtype,
                "storage_dtype": storage_dtype,
                "finite_count": int(arr.size),
                "nonfinite_count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "raw_bytes": int(arr.nbytes),
            })

        _admit_global("token_ids", np.asarray(ids, dtype="<i8"), "int64", "int64")
        _admit_global(
            "supervised_token_count",
            np.asarray(supervised_token_count, dtype="<i8"), "int64", "int64",
        )
        _admit_global(
            "causal_target_offset",
            np.asarray(causal_target_offset, dtype="<i8"), "int64", "int64",
        )
        _admit_global(
            "dataset_row", np.asarray(int(dataset_row), dtype="<i8"),
            "int64", "int64",
        )
        _admit_global(
            "optimizer_step", np.asarray(int(optimizer_step), dtype="<i8"),
            "int64", "int64",
        )
        loss_name = (
            "training_loss" if cap.phase == "pre_update" else "post_update_loss"
        )
        _admit_global(
            loss_name, np.float32(loss).reshape(()), "float32", "float32",
        )

        return tto.DerivedCapture(
            step=cap.step,
            phase=cap.phase,
            arrays=arrays,
            tensor_records=records,
            token_ids=ids,
            token_sha256=tto.token_ids_sha256(ids),
            sequence_length=seq_len if seq_len is not None else 0,
            raw_bytes=raw_bytes,
        )


# ---------------------------------------------------------------------------
# paired post-update replay
# ---------------------------------------------------------------------------


def post_update_replay(model, observer: TrainingLayerTransactionObserver,
                       step: int, inputs, targets, loss_fn):
    """Replay identical inputs under theta_step with capture, no side effects.

    RNG is snapshotted immediately before the replay and restored in finally.
    Gradients, parameters, optimizer state, and train/eval mode are untouched.
    Returns (post_loss_value, BoundaryCapture).
    """
    snap = tto.rng_snapshot()
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
        tto.rng_restore(snap)


# ---------------------------------------------------------------------------
# writing -- NPZ writer reused verbatim from Camera 1
# ---------------------------------------------------------------------------

write_step_npz = tto.write_step_npz
sha256_file = tto.sha256_file
token_ids_sha256 = tto.token_ids_sha256


def expected_filenames() -> list:
    out = []
    for step in range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1):
        for phase in PHASES:
            out.append(f"step_{step:06d}_{phase}.npz")
    return sorted(out)


def finalize_layer_transaction_dir(out_dir, file_records: list,
                                   run_meta: dict) -> dict:
    """Write storage_report.json, manifest.json, SHA256SUMS in the proven
    acyclic order: (a) NPZs (already written) -> (b) storage_report.json ->
    (c) hash NPZs + storage report -> (d) manifest.json embedding those hashes
    (never its own) -> (e) SHA256SUMS covering NPZs + storage_report.json +
    manifest.json (never itself)."""
    out_dir = Path(out_dir)
    wanted = expected_filenames()
    if len(file_records) != len(wanted):
        raise ObserverError(
            f"expected {len(wanted)} capture files, have {len(file_records)}"
        )
    file_records = sorted(file_records, key=lambda r: r["filename"])
    if [r["filename"] for r in file_records] != wanted:
        raise ObserverError(
            f"capture filenames {[r['filename'] for r in file_records]} != "
            f"expected {wanted}"
        )
    # each step must pair exactly one pre and one post file with identical
    # token identity
    by_step: dict = {}
    for r in file_records:
        by_step.setdefault(r["step"], {})[r["phase"]] = r
    for step, pair in by_step.items():
        if set(pair) != set(PHASES):
            raise ObserverError(
                f"step {step}: phases {sorted(pair)} != {sorted(PHASES)}"
            )
        if (
            pair["pre_update"]["token_ids_sha256"]
            != pair["post_update"]["token_ids_sha256"]
        ):
            raise ObserverError(
                f"step {step}: pre/post token_ids_sha256 mismatch; the replay "
                "must use identical token IDs"
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
        "lto_schema_version": LTO_SCHEMA_VERSION,
        "camera_number": CAMERA_NUMBER,
        "camera_family": CAMERA_FAMILY,
        "derivative_scope": "training_loss_vjp",
        "full_jacobian_captured": False,
        "gradient_target": "live block-boundary tensors",
        "gradient_stage": "pre_parameter_gradient_clipping",
        "gradient_semantics": "d_mean_cross_entropy_d_boundary",
        "descent_pressure_semantics": "negative_raw_gradient",
        "loss_reduction": "mean",
        "capture_window": sorted({r["step"] for r in file_records}),
        "phases": list(PHASES),
        "finite_assertions_passed": True,
        "files": [
            {**r, "sha256": hashes[r["filename"]]} for r in file_records
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
