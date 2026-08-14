"""Drop-in Camera 5: embedding learning."""

import training_embedding_learning_observer as telo


class Camera5:
    camera_id = 5
    family = "embedding_learning"

    def __init__(self, scene, out_dir):
        self.scene = scene
        self.out_dir = out_dir
        self.observer = telo.TrainingEmbeddingLearningObserver(
            scene.model, scene.optimizer
        )
        self.records = []
        self._cm = None
        self.cap = None

    def start_step(self, ctx):
        self._cm = self.observer.capture(ctx.step, ctx.input_ids, ctx.targets)
        self.cap = self._cm.__enter__()

    def before_clip(self, ctx):
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
        self.records.append(telo.write_step_npz(self.out_dir, derived))
        self.cap.release_snapshots()
        self.cap = None

    def close_step(self, ctx):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
        if self.observer.hooks_active():
            raise RuntimeError("Camera 5 hooks leaked")

    def finalize_run(self, run_meta):
        return telo.finalize_embedding_learning_dir(
            self.out_dir,
            self.records,
            {**run_meta, "camera_number": 5, "camera_family": self.family},
            expected_files=len(run_meta["steps"]),
        )


def create_camera(scene, out_dir):
    return Camera5(scene, out_dir)
