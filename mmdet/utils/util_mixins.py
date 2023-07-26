
import warnings


class NiceRepr(object):

    def __nice__(self):
        if hasattr(self, '__len__'):
            return str(len(self))
        else:
            raise NotImplementedError(
                f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)
