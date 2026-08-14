"""Drop-in Camera 3: optimizer delivery."""

import training_optimizer_delivery_observer as todo


class Camera3:
    camera_id = 3
    family = "optimizer_delivery"

    def __init__(self, scene, out_dir):
        self.scene = scene
        self.out_dir = out_dir
        # The observer re-derives the expected clip coefficient and cross-checks
        # it against what the trainer's clipper actually did, so it must hold
        # the SAME threshold.  Taken from the scene and never defaulted: its own
        # default is 1.0, and a stale threshold either fails every unclipped
        # step or, when the gradient is below both values, silently writes the
        # wrong clip_max_norm into the artifact.
        grad_clip = getattr(scene, "grad_clip", None)
        if grad_clip is None:
            raise RuntimeError(
                "Camera 3 needs scene.grad_clip: the observer's clip "
                "cross-check must police the threshold the trainer used, and "
                "guessing it corrupts clip_max_norm and clip_fired"
            )
        self.observer = todo.TrainingOptimizerDeliveryObserver(
            scene.model, scene.optimizer, max_grad_norm=float(grad_clip)
        )
        self.records = []
        self._cm = None
        self.cap = None

    def before_clip(self, ctx):
        self._cm = self.observer.capture(ctx.step)
        self.cap = self._cm.__enter__()
        self.observer.record_pre_clip(self.cap)

    def after_clip(self, ctx):
        self.observer.record_clip(self.cap, ctx.grad_norm)

    def after_optimizer_step(self, ctx):
        self.observer.record_post_step(self.cap)
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
        self.observer.assert_complete(self.cap)

    def finalize_step(self, ctx):
        derived = self.observer.derive(
            self.cap,
            ctx.ids,
            training_loss=ctx.loss_value,
            dataset_row=ctx.dataset_row,
            optimizer_step=ctx.step,
        )
        self.records.append(todo.write_step_npz(self.out_dir, derived))
        self.cap.release_snapshots()
        self.cap = None

    def close_step(self, ctx):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
        if self.observer.hooks_active():
            raise RuntimeError("Camera 3 unexpectedly registered hooks")

    def finalize_run(self, run_meta):
        return todo.finalize_optimizer_delivery_dir(
            self.out_dir,
            self.records,
            {**run_meta, "camera_number": 3, "camera_family": self.family},
            expected_files=len(run_meta["steps"]),
        )


def create_camera(scene, out_dir):
    return Camera3(scene, out_dir)
