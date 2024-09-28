from __future__ import annotations
import bitstring
import os

class Options:
    """Internal class to create singleton module options instance."""
    _instance = None

    def __init__(self):
        self.set_lsb0(False)
        self._bytealigned = False
        self.mxfp_overflow = 'saturate'
        self.no_color = False
        no_color = os.getenv('NO_COLOR')
        self.no_color = True if no_color else False

    def __repr__(self) -> str:
        attributes = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith('_') and (not callable(getattr(self, attr)))}
        return '\n'.join((f'{attr}: {value!r}' for attr, value in attributes.items()))

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Options, cls).__new__(cls)
        return cls._instance

class Colour:

    def __new__(cls, use_colour: bool) -> Colour:
        x = super().__new__(cls)
        if use_colour:
            cls.blue = '\x1b[34m'
            cls.purple = '\x1b[35m'
            cls.green = '\x1b[32m'
            cls.off = '\x1b[0m'
        else:
            cls.blue = cls.purple = cls.green = cls.off = ''
        return x