from pathlib import Path
import json
import os
import copy
import time
from ._model import *  # noqa

import logging

logger = logging.getLogger(__name__)


class MultiModel:
    """A class for running multiple Pywr model simultaneously.

    The `MultiModel` class allows running multiple `pywr.Model` simultaneously. The combined simulation runs
    the sub-models in an interleaved fashion. Each sub-model is progressed one time-step in a pre-determined order
    before moving on the next time-step.
    """

    def __init__(self):
        self.models = {}  # TODO this does depend on ordering here

    def add_model(self, name: str, model: Model):
        """Add a model."""
        self.models[name] = model
        model.parent = self

    @classmethod
    def loads(cls, data, model=None, path=None, solver=None, **kwargs):
        """Read JSON data from a string and parse it as a model document"""
        try:
            data = json.loads(data)
        except ValueError as e:
            message = e.args[0]
            if path:
                e.args = ("{} [{}]".format(e.args[0], os.path.basename(path)),)
            raise (e)

        return cls.load(data, model, path, solver, **kwargs)

    @classmethod
    def load(cls, data, model=None, path=None, solver=None, **kwargs):
        if isinstance(data, (str, Path)):
            # argument is a filename
            logger.info('Loading model from file: "{}"'.format(path))
            path = data
            with open(path, "r") as f:
                data = f.read()
            return cls.loads(data, model, path, solver)

        if hasattr(data, "read"):
            logger.info("Loading model from file-like object.")
            # argument is a file-like object
            data = data.read()
            return cls.loads(data, model, path, solver)

        return cls._load_from_dict(data, model=model, path=path, solver=None, **kwargs)

    @classmethod
    def _load_from_dict(cls, data, model=None, path=None, solver=None, **kwargs):
        """Load data from a dictionary."""
        # data is a dictionary, make a copy to avoid modify the input
        data = copy.deepcopy(data)

        if model is None:
            model = cls()

        # Load the sub-models
        sub_model_paths = data["models"]
        for sub_model_definition in sub_model_paths:
            sub_model_name = sub_model_definition["name"]
            sub_model_path = sub_model_definition["path"]
            if path is not None:
                sub_model_path = os.path.join(os.path.dirname(path), sub_model_path)

            sub_model = Model.load(sub_model_path, path=path, solver=solver)
            model.add_model(sub_model_name, sub_model)

        return model

    @property
    def dirty(self) -> bool:
        """Returns true if any of the models are 'dirty'."""
        return any(
            model.dirty or model.timestepper.dirty for model in self.models.values()
        )

    def setup(self):
        for model in self.models.values():
            model.setup()

    def reset(self):
        for model in self.models.values():
            model.reset()

    # def _step(self):
    #     for model in self.models.values():
    #         model.before()
    #     # solve the current timestep for all models
    #     rets = []
    #     for model in self.models.values():
    #         ret = model.solve()
    #         rets.append(ret)
    #
    #     for model in self.models.values():
    #         model.after()
    #     return rets

    def _step(self):
        rets = []
        for model in self.models.values():
            model.before()
            ret = model.solve()
            rets.append(ret)
            model.after()
        return rets

    def run(self):
        """Run the multi model simulation"""
        logger.info("Start multi model run ...")
        t0 = time.time()
        timestep = None

        # Get each model's timestepper
        timesteppers = [m.timestepper for m in self.models.values()]

        try:
            if self.dirty:
                self.setup()
            else:
                self.reset()
            t1 = time.time()

            # Begin time-stepping
            for timesteps in zip(*timesteppers):
                # TODO assert these are all the same date
                # Check all timesteps are the same
                for model, timestep in zip(self.models.values(), timesteps):
                    model.timestep = timestep
                # Perform the internal timestep
                rets = self._step()
            t2 = time.time()
        finally:
            for model in self.models.values():
                model.finish()
        t3 = time.time()
        return
