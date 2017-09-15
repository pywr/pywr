from pywr.recorders import Recorder
import time

class ProgressRecorder(Recorder):
    """Simple text-based progress notifications"""
    def setup(self):
        self.last_progress = -1
        self.last_timestep = 0

    def reset(self):
        self.t0 = time.time()
        self.combinations = len(self.model.scenarios.combinations)

    def after(self):
        total_timesteps = len(self.model.timestepper)
        timestep = self.model.timestepper.current.index
        progress = int((timestep+1) / total_timesteps * 100)
        if progress > self.last_progress:
            self.last_progress = progress
            if progress >= 1:
                time_taken = time.time() - self.t0
                try:
                    speed = ((timestep-self.last_timestep)*self.combinations) / time_taken
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
            print("Completed {}%, {:.0f} steps/second".format(progress, speed))
        else:
            print("Completed {}%".format(progress))

class JupyterProgressRecorder(ProgressRecorder):
    """Graphical progress bar for use in Jupyter notebooks"""
    def __init__(self, *args, **kwargs):
        super(JupyterProgressRecorder, self).__init__(*args, **kwargs)

    def reset(self):
        from ipywidgets import FloatProgress
        from IPython.display import display
        super(JupyterProgressRecorder, self).reset()
        self.progress_bar = FloatProgress(min=0, max=100, description='Running:')
        display(self.progress_bar)

    def update_progress(self, progress, speed):
        self.progress_bar.value = progress

    def finish(self):
        super(JupyterProgressRecorder, self).finish()
        if self.progress_bar.value >= 100.0:
            self.progress_bar.bar_style = "success"
        else:
            self.progress_bar.bar_style = "danger"

