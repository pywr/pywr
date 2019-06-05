

def parameter_property(private_name, wrap_constants=False):
    """Factory for parameter properties

    This property handles adding/removing parameters from the tree.

    Usage
    =====

    class NewParameter(Parameter):
        control_curve = parameter_property("_control_curve")
    """
    def parameter():
        def fget(self):
            return getattr(self, private_name)
        def fset(self, parameter):
            old_parameter = getattr(self, private_name)
            if old_parameter:
                self.children.remove(old_parameter)
            if wrap_constants and isinstance(parameter, (float, int)):
                from pywr.parameters import ConstantParameter
                parameter = ConstantParameter(self.model, value=parameter)
            self.children.add(parameter)
            setattr(self, private_name, parameter)
        return {"fget": fget, "fset": fset}
    parameter = property(**parameter())
    return parameter
