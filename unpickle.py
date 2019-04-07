import pickle


class RestrictedUnpickler(pickle.Unpickler):
    safe_builtins = frozenset({
        'int',
        'float',
        'str',
        'tuple',
    })

    def find_class(self, module, name):
        # Only allow safe classes from builtins.
        if module == "builtins" and name in self.safe_builtins:
            return getattr(builtins, name)

        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" % (module, name))
