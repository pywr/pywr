from pywr.parameters import ConstantParameter


class MyParameter(ConstantParameter):
    pass


MyParameter.register()
