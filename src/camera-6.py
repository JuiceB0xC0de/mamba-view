"""Drop-in Camera 6: layer transaction."""

import torch.nn.functional as F

import training_layer_transaction_observer as lto


class Camera6:
    camera_id = 6
    family = "layer_transaction"

    def __init__(self, scene, out_dir):
        self.scene = scene
        self.out_dir = out_dir
        self.observer = lto.TrainingLayerTransactionObserver(scene.model)
        self.records = []
        self._cm = None
        self.pre = self.post = None
        self.post_loss = None

    def before_forward(self, ctx):
        self._cm = self.observer.capture(ctx.step, "pre_update")
        self.pre = self._cm.__enter__()

    def after_backward(self, ctx):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
        self.observer.check_capture_complete(self.pre)

    def post_update(self, ctx):
        def ce(output, targets):
            logits = output.logits.float()
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )

        self.post_loss, self.post = lto.post_update_replay(
            self.scene.model,
            self.observer,
            ctx.step,
            ctx.input_ids,
            ctx.targets,
            ce,
        )

    def finalize_step(self, ctx):
        pre = self.observer.derive(
            self.pre,
            ctx.ids,
            loss=ctx.loss_value,
            dataset_row=ctx.dataset_row,
            optimizer_step=ctx.step,
        )
        post = self.observer.derive(
            self.post,
            ctx.ids,
            loss=self.post_loss,
            dataset_row=ctx.dataset_row,
            optimizer_step=ctx.step,
        )
        self.records.extend(
            [
                lto.write_step_npz(self.out_dir, pre),
                lto.write_step_npz(self.out_dir, post),
            ]
        )
        self.pre = self.post = None

    def close_step(self, ctx):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
        if self.observer.hooks_active():
            raise RuntimeError("Camera 6 hooks leaked")

    def finalize_run(self, run_meta):
        return lto.finalize_layer_transaction_dir(
            self.out_dir,
            self.records,
            {**run_meta, "camera_number": 6, "camera_family": self.family},
        )


def create_camera(scene, out_dir):
    return Camera6(scene, out_dir)
