from . import nodes


def validate_with(validation_klass):
    """ Decorator for """
    def validation_decorator(klass):
        klass.schema = validation_klass()
        return klass
    return validation_decorator




