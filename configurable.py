import importlib
import inspect


def classpath_of(obj) -> str:
    if inspect.isfunction(obj):
        return module_path_of(obj)
    return "{0}.{1}".format(obj.__class__.__module__, obj.__class__.__name__)
    
def module_path_of(func) -> str:
    return inspect.getmodule(func).__name__ + '.' + func.__name__

def str_to_type(classpath):
    module_name, class_name = classpath.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls

class Configurable(object):
    @staticmethod
    def from_config(config: dict, **kwargs):
        """Build an object from config.

        Args:
          config: A ``dict`` holding parameters for its constructor. It has to contain a `classpath` key,
                    which has a classpath str as its value. ``classpath`` will determine the type of object
                    being deserialized.
          kwargs: Arguments not used.

        Returns: A deserialized object.

        """
        cls = config.get('classpath', None)
        assert cls, f'{config} doesn\'t contain classpath field'
        cls = str_to_type(cls)
        deserialized_config = dict(config)
        for k, v in config.items():
            if isinstance(v, dict) and 'classpath' in v:
                deserialized_config[k] = Configurable.from_config(v)
        if cls.from_config == Configurable.from_config:
            deserialized_config.pop('classpath')
            return cls(**deserialized_config)
        else:
            return cls.from_config(deserialized_config)