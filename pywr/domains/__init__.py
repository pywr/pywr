from pywr import _core


class Domain(_core.Domain):
    def __init__(self, name="default", **kwargs):
        super(Domain, self).__init__(name)
        self.color = kwargs.pop("color", "#FF6600")
