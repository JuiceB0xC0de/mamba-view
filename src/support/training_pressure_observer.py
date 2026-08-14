"""Training-loss Pressure Map observer for Mamba-3 (Camera 2, steps 20-25).

Modal-free by design so every mechanism is CPU-testable. The Modal trainer
(modal_pressure_observer.py) imports this module; nothing here imports modal,
wandb, or CUDA-only code.

Camera 2 observes ONE training-loss VJP at the raw control surface: during the
real training forward and backward it records the gradient of the actual mean
next-token cross-entropy with respect to the four raw regions of every layer's
live ``mixer.in_proj`` output -- dd_dt, dd_A, trap, angles.

    grad_dd_dt     = d(mean_cross_entropy) / d(dd_dt)
    grad_dd_A      = d(mean_cross_entropy) / d(dd_A)
    grad_trap_raw  = d(mean_cross_entropy) / d(trap_raw)
    grad_angle_raw = d(mean_cross_entropy) / d(angle_raw)

Stored gradients are the raw autograd values, exactly. They are observed
BEFORE parameter-gradient clipping, are NOT applied gradients, NOT parameter
gradients, and NOT the full Jacobian. The offline analyzer may additionally
report descent_pressure = -gradient; the stored gradient is never renamed.

This camera is one component of the larger pre-canary observer suite. It is
not a full Jacobian camera and does not complete the microscope.

Infrastructure reuse: mixer discovery, per-layer ``InProjSpec`` geometry,
deterministic NPZ writing, tensor records, token hashing, and non-finite
policy all come from ``training_time_observer`` (Camera 1, untouched). No
second in-projection geometry implementation exists here.
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
# this import; Camera 2 reuses Camera 1's discovery, spec, writer, and errors.
import training_time_observer as tto  # noqa: E402

ObserverError = tto.ObserverError
NonFiniteCaptureError = tto.NonFiniteCaptureError
GeometryMismatchError = tto.GeometryMismatchError

PRESSURE_SCHEMA_VERSION = 1
CAMERA_NUMBER = 2
CAMERA_FAMILY = "pressure_map"
CAPTURE_START_STEP = 20
CAPTURE_END_STEP = 25
CAPTURE_WINDOW = frozenset(range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1))
PHASE = "pressure"  # single phase; DerivedCapture.filename -> step_XXX_pressure.npz

# in_proj slice names (spec bounds) and their storage names.
HOOKED_SLICES = ("dd_dt", "dd_A", "trap", "angles")
_SLICE_TO_RAW = {
    "dd_dt": "dd_dt",
    "dd_A": "dd_A",
    "trap": "trap_raw",
    "angles": "angle_raw",
}
_SLICE_TO_GRAD = {
    "dd_dt": "grad_dd_dt",
    "dd_A": "grad_dd_A",
    "trap": "grad_trap_raw",
    "angles": "grad_angle_raw",
}

RAW_NAMES = ("dd_dt", "dd_A", "trap_raw", "angle_raw", "dt_bias", "A_floor")
GRAD_NAMES = ("grad_dd_dt", "grad_dd_A", "grad_trap_raw", "grad_angle_raw")
GLOBAL_NAMES = (
    "token_ids", "training_loss", "supervised_token_count",
    "dataset_row", "optimizer_step",
)

# Head-resolved gradient controls vs the shared pre-head-expansion angle
# gradient (aggregated downstream contributions; NEVER labeled per-head).
HEAD_GRAD_NAMES = ("grad_dd_dt", "grad_dd_A", "grad_trap_raw")
ANGLE_GRAD_NAME = "grad_angle_raw"


@dataclass
class PressureCapture:
    """Raw forward slices + backward gradient slices for one captured step."""

    step: int
    raw: dict = field(default_factory=dict)            # layer idx -> {slice: tensor}
    grads: dict = field(default_factory=dict)          # layer idx -> {slice: tensor}
    source_dtypes: dict = field(default_factory=dict)  # layer idx -> str (forward)
    grad_source_dtypes: dict = field(default_factory=dict)
    forward_shapes: dict = field(default_factory=dict)  # layer idx -> tuple
    dt_bias: dict = field(default_factory=dict)
    dt_bias_source_dtype: dict = field(default_factory=dict)
    a_floor: dict = field(default_factory=dict)


class TrainingPressureObserver:
    """Camera 2: temporary forward hooks + per-step tensor gradient hooks.

    Module forward hooks exist only inside an active ``capture`` context.
    Each forward hook registers a tensor gradient hook on the ACTUAL live
    in_proj output tensor (never on newly created slice views); the tensor
    hook slices the four canonical regions from the accumulated gradient,
    detaches compact clones, and returns None -- the gradient stream is never
    replaced or altered. All hooks are removed in ``finally``, surviving
    forward, loss, and backward exceptions. ``retain_grad()`` is never used
    and the full output gradient is never retained.
    """

    def __init__(self, model: torch.nn.Module, capture_steps=None):
        # Defaults bind at import; resolve here so the window can be retargeted.
        capture_steps = CAPTURE_WINDOW if capture_steps is None else capture_steps
        # Reuse Camera 1's constructor wholesale: mixer discovery, per-layer
        # InProjSpec, cross-layer geometry assertion, A_floor presence.
        base = tto.TimeRetentionObserver(model, capture_steps)
        self.capture_steps = base.capture_steps
        self.layers = base.layers
        self.specs = base.specs
        self.geometry = base.geometry

    def should_capture(self, step: int) -> bool:
        return int(step) in self.capture_steps

    def module_hooks_active(self) -> bool:
        return any(
            len(mixer.in_proj._forward_hooks) > 0 for _, mixer in self.layers
        )

    # Alias used by trainer-side leak assertions, mirroring Camera 1's name.
    hooks_active = module_hooks_active

    def _make_forward_hook(self, layer_idx: int, spec, cap: PressureCapture,
                           tensor_handles: list):
        expected_width = int(spec.bounds["angles"][1])

        def hook(module, args, output):
            if layer_idx in cap.raw:
                raise ObserverError(
                    f"layer {layer_idx} observed twice within one capture "
                    f"(step {cap.step}); unexpected forward re-entry"
                )
            if not isinstance(output, torch.Tensor):
                raise ObserverError(
                    f"layer {layer_idx}: in_proj output is not a tensor "
                    f"({type(output).__name__})"
                )
            if not output.requires_grad:
                raise ObserverError(
                    f"layer {layer_idx} step {cap.step}: in_proj output does "
                    "not require grad; Camera 2 must observe the real "
                    "training forward, not a no-grad forward"
                )
            if int(output.shape[-1]) != expected_width:
                raise GeometryMismatchError(
                    f"layer {layer_idx}: in_proj output width "
                    f"{int(output.shape[-1])} != spec width {expected_width}"
                )

            det = output.detach()
            slices = {}
            for k in HOOKED_SLICES:
                lo, hi = spec.bounds[k]
                # clone() releases the full projection storage.
                slices[k] = det[..., lo:hi].clone()
            cap.raw[layer_idx] = slices
            cap.source_dtypes[layer_idx] = str(output.dtype)
            cap.forward_shapes[layer_idx] = tuple(output.shape)

            expected_shape = tuple(output.shape)

            def grad_hook(grad):
                if layer_idx in cap.grads:
                    raise ObserverError(
                        f"layer {layer_idx} step {cap.step}: gradient hook "
                        "fired twice; unexpected second backward delivery"
                    )
                if tuple(grad.shape) != expected_shape:
                    raise ObserverError(
                        f"layer {layer_idx} step {cap.step}: gradient shape "
                        f"{tuple(grad.shape)} != forward output shape "
                        f"{expected_shape}"
                    )
                g = grad.detach()
                gslices = {}
                for k in HOOKED_SLICES:
                    lo, hi = spec.bounds[k]
                    gslices[k] = g[..., lo:hi].clone()
                cap.grads[layer_idx] = gslices
                cap.grad_source_dtypes[layer_idx] = str(grad.dtype)
                # Mandatory: never replace or alter the gradient.
                return None

            # Hook the ACTUAL full live output tensor used downstream.
            tensor_handles.append(output.register_hook(grad_hook))
            return None

        return hook

    @contextmanager
    def capture(self, step: int):
        """Spans forward -> loss -> loss.backward(); exit before clipping.

        The trainer owns the single ordinary backward call; this observer
        never calls backward itself.
        """
        if not self.should_capture(step):
            raise ObserverError(
                f"capture requested for step {step}, outside window "
                f"{sorted(self.capture_steps)}"
            )
        cap = PressureCapture(step=int(step))
        for idx, (_, mixer) in enumerate(self.layers):
            cap.dt_bias[idx] = mixer.dt_bias.detach().float().clone()
            cap.dt_bias_source_dtype[idx] = str(mixer.dt_bias.dtype)
            cap.a_floor[idx] = float(mixer.A_floor)
        module_handles = []
        tensor_handles: list = []
        try:
            for idx, (_, mixer) in enumerate(self.layers):
                module_handles.append(
                    mixer.in_proj.register_forward_hook(
                        self._make_forward_hook(
                            idx, self.specs[idx], cap, tensor_handles
                        )
                    )
                )
            yield cap
        finally:
            for h in module_handles:
                h.remove()
            for h in tensor_handles:
                h.remove()

    # ------------------------------------------------------------------
    # finalization
    # ------------------------------------------------------------------

    def assert_complete(self, cap: PressureCapture) -> None:
        """In-memory completeness check, run before clip/step: every layer
        must have delivered both its forward slices and its backward
        gradient. Missing anything fails loudly here."""
        expected = set(range(len(self.layers)))
        if set(cap.raw.keys()) != expected:
            raise ObserverError(
                f"step {cap.step}: forward captured layers "
                f"{sorted(cap.raw)} != expected {sorted(expected)}"
            )
        if set(cap.grads.keys()) != expected:
            missing = sorted(expected - set(cap.grads.keys()))
            raise ObserverError(
                f"step {cap.step}: no gradient delivered for layers "
                f"{missing}; backward missing or incomplete"
            )

    def derive(self, cap: PressureCapture, token_ids, *, training_loss: float,
               dataset_row: int, optimizer_step: int) -> tto.DerivedCapture:
        """Package raw context + raw gradients into a DerivedCapture.

        Raises NonFiniteCaptureError on ANY non-finite raw activation,
        captured gradient, or loss. The stored gradient is the gradient of
        the MEAN cross-entropy loss; it is never rescaled here.
        """
        self.assert_complete(cap)

        if not np.isfinite(training_loss):
            raise NonFiniteCaptureError(
                f"step {cap.step}: non-finite training loss {training_loss}"
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
                    f"step {cap.step} layer {layer_idx} tensor {name}: "
                    f"{total - finite} non-finite of {total}"
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

        def _flat(layer_idx, group, slices, name):
            t = slices[name]
            if t.ndim != 3 or t.shape[0] != 1:
                raise ObserverError(
                    f"step {cap.step} layer {layer_idx} {group} slice {name}: "
                    f"expected shape (1, T, width), got {tuple(t.shape)}"
                )
            return t[0].float()

        for idx in range(len(self.layers)):
            src_dtype = cap.source_dtypes[idx]
            grad_dtype = cap.grad_source_dtypes[idx]

            for k in HOOKED_SLICES:
                fwd = _flat(idx, "forward", cap.raw[idx], k)
                grd = _flat(idx, "gradient", cap.grads[idx], k)
                if tuple(grd.shape) != tuple(fwd.shape):
                    raise ObserverError(
                        f"step {cap.step} layer {idx} slice {k}: gradient "
                        f"shape {tuple(grd.shape)} != forward shape "
                        f"{tuple(fwd.shape)}"
                    )
                if seq_len is None:
                    seq_len = int(fwd.shape[0])
                _admit(idx, _SLICE_TO_RAW[k], fwd, src_dtype)
                _admit(idx, _SLICE_TO_GRAD[k], grd, grad_dtype)

            _admit(idx, "dt_bias", cap.dt_bias[idx], cap.dt_bias_source_dtype[idx])
            # 0-d scalar; manifest records shape [].
            _admit(idx, "A_floor", torch.tensor(cap.a_floor[idx]), "python_float")

        ids = [int(i) for i in token_ids]
        supervised_token_count = len(ids) - 1  # targets = ids[1:]
        if seq_len is not None and supervised_token_count != seq_len:
            raise ObserverError(
                f"step {cap.step}: supervised_token_count "
                f"{supervised_token_count} != captured time dimension {seq_len}"
            )

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
            "training_loss", np.float32(training_loss).reshape(()),
            "float32", "float32",
        )
        _admit_global(
            "supervised_token_count",
            np.asarray(supervised_token_count, dtype="<i8"), "int64", "int64",
        )
        _admit_global(
            "dataset_row", np.asarray(int(dataset_row), dtype="<i8"),
            "int64", "int64",
        )
        _admit_global(
            "optimizer_step", np.asarray(int(optimizer_step), dtype="<i8"),
            "int64", "int64",
        )

        return tto.DerivedCapture(
            step=cap.step,
            phase=PHASE,
            arrays=arrays,
            tensor_records=records,
            token_ids=ids,
            token_sha256=tto.token_ids_sha256(ids),
            sequence_length=seq_len if seq_len is not None else 0,
            raw_bytes=raw_bytes,
        )


# ---------------------------------------------------------------------------
# compact W&B scalar summaries (no tensor arrays in history)
# ---------------------------------------------------------------------------


def gradient_scalars(derived: tto.DerivedCapture, num_layers: int) -> dict:
    """L2 norm per gradient control across all layers, plus overall max |g|."""
    out = {}
    overall_max = 0.0
    for name in GRAD_NAMES:
        sq = 0.0
        for idx in range(num_layers):
            g = derived.arrays[f"layer_{idx:02d}/{name}"]
            sq += float(np.sum(np.square(g, dtype=np.float64)))
            overall_max = max(overall_max, float(np.max(np.abs(g))))
        out[f"pressure/{name}_l2"] = float(np.sqrt(sq))
    out["pressure/overall_max_abs"] = overall_max
    out["pressure/supervised_token_count"] = int(
        derived.arrays["supervised_token_count"]
    )
    return out


# ---------------------------------------------------------------------------
# writing -- NPZ writer reused verbatim from Camera 1
# ---------------------------------------------------------------------------

write_step_npz = tto.write_step_npz
sha256_file = tto.sha256_file
token_ids_sha256 = tto.token_ids_sha256


def expected_filenames() -> list:
    return [
        f"step_{step:06d}_{PHASE}.npz"
        for step in range(CAPTURE_START_STEP, CAPTURE_END_STEP + 1)
    ]


def finalize_pressure_dir(out_dir, file_records: list, run_meta: dict,
                          expected_files: int = 6) -> dict:
    """Write storage_report.json, manifest.json, SHA256SUMS in the proven
    acyclic order: (a) NPZs (already written) -> (b) storage_report.json ->
    (c) hash NPZs + storage report -> (d) manifest.json embedding those hashes
    (never its own) -> (e) SHA256SUMS covering NPZs + storage_report.json +
    manifest.json (never itself)."""
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
        "pressure_schema_version": PRESSURE_SCHEMA_VERSION,
        "camera_number": CAMERA_NUMBER,
        "camera_family": CAMERA_FAMILY,
        "derivative_scope": "training_loss_vjp",
        "full_jacobian_captured": False,
        "gradient_target": "mixer.in_proj output",
        "gradient_stage": "pre_parameter_gradient_clipping",
        "gradient_semantics": "d_mean_cross_entropy_d_raw_control",
        "descent_pressure_semantics": "negative_raw_gradient",
        "loss_reduction": "mean",
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
