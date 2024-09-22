# utils.py

import copyreg
import types


def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__class__
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    func = None
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    if func is None:
        raise AttributeError(f"Method {func_name} not found")
    return func.__get__(obj, cls)


# Register the pickle functions
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
