"""Drop-in Camera 2: pressure map."""

import training_pressure_observer as tpo


class Camera2:
    camera_id = 2
    family = "pressure_map"

    def __init__(self, scene, out_dir):
        self.scene = scene
        self.out_dir = out_dir
        self.observer = tpo.TrainingPressureObserver(scene.model)
        self.records = []
        self._cm = None
        self.cap = None

    def before_forward(self, ctx):
        self._cm = self.observer.capture(ctx.step)
        self.cap = self._cm.__enter__()

    def after_backward(self, ctx):
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
        self.records.append(tpo.write_step_npz(self.out_dir, derived))
        self.cap = None

    def close_step(self, ctx):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
        if self.observer.hooks_active():
            raise RuntimeError("Camera 2 hooks leaked")

    def finalize_run(self, run_meta):
        return tpo.finalize_pressure_dir(
            self.out_dir,
            self.records,
            {**run_meta, "camera_number": 2, "camera_family": self.family},
            expected_files=len(run_meta["steps"]),
        )


def create_camera(scene, out_dir):
    return Camera2(scene, out_dir)
