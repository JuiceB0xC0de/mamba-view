"""Drop-in Camera 1: time and retention."""

import torch.nn.functional as F

import training_time_observer as tto


class Camera1:
    camera_id = 1
    family = "time_retention"

    def __init__(self, scene, out_dir):
        self.scene = scene
        self.out_dir = out_dir
        self.observer = tto.TimeRetentionObserver(scene.model)
        self.records = []
        self._cm = None
        self.pre = None
        self.post = None
        self.post_loss = None

    def before_forward(self, ctx):
        self._cm = self.observer.capture(ctx.step, "pre_update")
        self.pre = self._cm.__enter__()

    def after_backward(self, ctx):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None

    def post_update(self, ctx):
        def ce(output, targets):
            logits = output.logits.float()
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )

        self.post_loss, self.post = tto.post_update_replay(
            self.scene.model,
            self.observer,
            ctx.step,
            ctx.input_ids,
            ctx.targets,
            ce,
        )

    def finalize_step(self, ctx):
        pre = self.observer.derive(self.pre, ctx.ids)
        post = self.observer.derive(self.post, ctx.ids)
        self.records.extend(
            [
                tto.write_step_npz(self.out_dir, pre),
                tto.write_step_npz(self.out_dir, post),
            ]
        )
        self.pre = self.post = None

    def close_step(self, ctx):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
        if self.observer.hooks_active():
            raise RuntimeError("Camera 1 hooks leaked")

    def finalize_run(self, run_meta):
        return tto.finalize_artifact_dir(
            self.out_dir,
            self.records,
            {**run_meta, "camera_number": 1, "camera_family": self.family},
            expected_files=2 * len(run_meta["steps"]),
        )


def create_camera(scene, out_dir):
    return Camera1(scene, out_dir)
