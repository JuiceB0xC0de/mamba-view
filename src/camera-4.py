"""Drop-in Camera 4: memory path."""

import torch
import torch.nn.functional as F

import training_memory_path_camera as tmpc


class Camera4:
    camera_id = 4
    family = "memory_path"

    def __init__(self, scene, out_dir):
        self.scene = scene
        self.out_dir = out_dir
        self.camera = tmpc.TrainingMemoryPathCamera(scene.model, scene.optimizer)
        self.adapter_report = self.camera.preflight_decode_shape_adapter()
        self.records = []
        self._capture_cm = None
        self._prefill_cm = None
        self.cap = self.pre = self.post = None

    def start_step(self, ctx):
        self._capture_cm = self.camera.capture(ctx.step, ctx.input_ids, ctx.targets)
        self.cap = self._capture_cm.__enter__()
        self.pre = self.camera.begin_phase(self.cap, "pre_update")
        fingerprint = tmpc.TrainerStateFingerprint(
            self.scene.model, self.scene.optimizer
        )
        (
            self.pre.cache_arrays,
            self.pre.cache_retained_bytes,
        ) = self.camera.record_cache_summary(
            ctx.input_ids, fingerprint, "pre_update cache replay"
        )

    def before_forward(self, ctx):
        self._prefill_cm = self.camera.prefill_hooks(self.pre.boundary, ctx.n_tokens)
        self._prefill_cm.__enter__()

    def after_forward(self, ctx):
        if self._prefill_cm is not None:
            self._prefill_cm.__exit__(None, None, None)
            self._prefill_cm = None
        self.camera.assert_boundary_complete(self.pre.boundary)

    def after_backward(self, ctx):
        self.pre.loss = ctx.loss_value
        self.camera.snapshot_parameters(self.pre)

    def post_update(self, ctx):
        self.post = self.camera.begin_phase(self.cap, "post_update")
        fingerprint = tmpc.TrainerStateFingerprint(
            self.scene.model, self.scene.optimizer
        )
        rng = tmpc.tmpo.rng_snapshot()
        try:
            with torch.no_grad():
                with self.camera.prefill_hooks(self.post.boundary, ctx.n_tokens):
                    logits = self.scene.model(ctx.input_ids).logits.float()
                self.post.loss = float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        ctx.targets.reshape(-1),
                    )
                )
        finally:
            tmpc.tmpo.rng_restore(rng)
        self.camera.assert_boundary_complete(self.post.boundary)
        fingerprint.assert_unchanged("Camera 4 post_update prefill")
        (
            self.post.cache_arrays,
            self.post.cache_retained_bytes,
        ) = self.camera.record_cache_summary(
            ctx.input_ids, fingerprint, "post_update cache replay"
        )
        self.camera.snapshot_parameters(self.post)
        self.camera.derive_reference(self.pre)
        self.camera.derive_reference(self.post)
        self.camera.assert_complete(self.cap)

    def finalize_step(self, ctx):
        for phase in tmpc.PHASES:
            derived = self.camera.derive(
                self.cap,
                phase,
                ctx.ids,
                dataset_row=ctx.dataset_row,
                optimizer_step=ctx.step,
            )
            self.records.append(tmpc.write_step_npz(self.out_dir, derived))

    def close_step(self, ctx):
        if self._prefill_cm is not None:
            self._prefill_cm.__exit__(None, None, None)
            self._prefill_cm = None
        if self._capture_cm is not None:
            self._capture_cm.__exit__(None, None, None)
            self._capture_cm = None
        self.cap = self.pre = self.post = None
        if self.camera.hooks_active():
            raise RuntimeError("Camera 4 hooks leaked")

    def finalize_run(self, run_meta):
        return tmpc.finalize_memory_path_dir(
            self.out_dir,
            self.records,
            {
                **run_meta,
                "camera_number": 4,
                "camera_family": self.family,
                "state_claims_enabled": False,
                "decode_shape_adapter": self.adapter_report,
            },
            expected_files=2 * len(run_meta["steps"]),
        )


def create_camera(scene, out_dir):
    return Camera4(scene, out_dir)
