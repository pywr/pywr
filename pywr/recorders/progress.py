from pywr.recorders import Recorder
import time
import logging

logger = logging.getLogger(__name__)


class ProgressRecorder(Recorder):
    """Simple text-based progress notifications

    Parameters
    ----------
    print_func : callable or None
        The function to call when updating progress. The function is given a single str argument that
        contains the progress message. Defaults to `logger.info`.
    """

    def __init__(self, *args, **kwargs):
        print_func = kwargs.pop("print_func", None)
        super(ProgressRecorder, self).__init__(*args, **kwargs)
        if print_func is None:
            print_func = logger.info
        self.print_func = print_func
        self.last_progress = None
        self.last_timestep = None
        self.t0 = None
        self.combinations = None

    def reset(self):
        self.last_progress = -1
        self.last_timestep = 0
        self.t0 = time.time()
        self.combinations = len(self.model.scenarios.combinations)

    def after(self):
        total_timesteps = len(self.model.timestepper)
        timestep = self.model.timestepper.current.index
        progress = int((timestep + 1) / total_timesteps * 100)
        if progress > self.last_progress:
            self.last_progress = progress
            if progress >= 1:
                time_taken = time.time() - self.t0
                try:
                    speed = (
                        (timestep - self.last_timestep) * self.combinations
                    ) / time_taken
                except ZeroDivisionError:
                    speed = float("inf")
                self.last_timestep = timestep
                self.update_progress(progress, speed)
            else:
                self.last_timestep = timestep
                self.update_progress(progress, None)
            self.t0 = time.time()
            self.last_timestep = timestep

    def update_progress(self, progress, speed=None):
        if speed is not None:
            self.print_func(
                "Completed {}%, {:.0f} steps/second".format(progress, speed)
            )
        else:
            self.print_func("Completed {}%".format(progress))


class JupyterProgressRecorder(ProgressRecorder):
    """Graphical progress bar for use in Jupyter notebooks"""

    def __init__(self, *args, **kwargs):
        super(JupyterProgressRecorder, self).__init__(*args, **kwargs)

    def reset(self):
        from ipywidgets import FloatProgress, HBox, Label, Layout
        from IPython.display import display

        super(JupyterProgressRecorder, self).reset()
        self.progress_bar = FloatProgress(min=0, max=100, description="Running:")
        self.label = Label("", layout=Layout(width="100%"))
        self.box = HBox([self.progress_bar, self.label])
        display(self.box)

    def update_progress(self, progress, speed):
        self.progress_bar.value = progress
        if speed is not None:
            self.label.value = "{:.0f} steps/second".format(speed)
        else:
            self.label.value = ""

    def finish(self):
        super(JupyterProgressRecorder, self).finish()
        if self.progress_bar.value >= 100.0:
            self.progress_bar.bar_style = "success"
        else:
            self.progress_bar.bar_style = "danger"
