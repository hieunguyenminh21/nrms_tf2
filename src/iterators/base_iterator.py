def abstractmethod(funcobj):
    """A decorator indicating abstract methods.

    Requires that the metaclass is ABCMeta or derived from it.  A
    class that has a metaclass derived from ABCMeta cannot be
    instantiated unless all of its abstract methods are overridden.
    The abstract methods can be called using any of the normal
    'super' call mechanisms.  abstractmethod() may be used to declare
    abstract methods for properties and descriptors.

    Usage:

        class C(metaclass=ABCMeta):
            @abstractmethod
            def my_abstract_method(self, ...):
                ...
    """
    funcobj.__isabstractmethod__ = True
    return funcobj


class BaseIterator(object):
    """Abstract base iterator class"""

    @abstractmethod
    def parser_one_line(self, line):
        """Abstract method. Parse one string line into feature values.

        Args:
            line (str): A string indicating one instance.
        """
        pass

    @abstractmethod
    def load_data_from_file(self, infile):
        """Abstract method. Read and parse data from a file.

        Args:
            infile (str): Text input file. Each line in this file is an instance.
        """
        pass

    @abstractmethod
    def _convert_data(self, labels, features):
        pass

    @abstractmethod
    def gen_feed_dict(self, data_dict):
        """Abstract method. Construct a dictionary that maps graph elements to values.

        Args:
            data_dict (dict): A dictionary that maps string name to numpy arrays.
        """
        pass
