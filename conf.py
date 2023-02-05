import functools
import inspect

global_var = {"wrapped": False}


def configure(fun_or_class, **kwargs):
    if inspect.isclass(fun_or_class):
        for k, f in inspect.getmembers(fun_or_class, inspect.isroutine):
            setattr(fun_or_class, k, configure(f))
        return fun_or_class
    else:

        @functools.wraps(fun_or_class)
        def wrapped(*args, **kwargs):
            global_var["wrapped"] = True
            outputs = fun_or_class(*args, **kwargs)
            global_var["wrapped"] = False
            return outputs

        return wrapped


class Print:
    def __init__(self):
        print("Initialize:", global_var)

    def __call__(self):
        print("Called:", global_var)


f = Print()
f()
configure(f)()
f()
