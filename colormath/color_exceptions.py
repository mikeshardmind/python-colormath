"""
This module contains exceptions for use throughout the L11 Colorlib.
"""

from typing import Any


class ColorMathException(Exception):
    """
    Base exception for all colormath exceptions.
    """


class UndefinedConversionError(ColorMathException):
    """
    Raised when the user asks for a color space conversion that does not exist.
    """

    def __init__(self, cobj: object, cs_to: object):
        super().__init__(cobj, cs_to)
        self.message = f"Conversion from {cobj} to {cs_to} is not defined."


class InvalidIlluminantError(ColorMathException):
    """
    Raised when an invalid illuminant is set on a ColorObj.
    """

    def __init__(self, illuminant: object):
        super().__init__(illuminant)
        self.message = f"Invalid illuminant specified: {illuminant}"


class InvalidObserverError(ColorMathException):
    """
    Raised when an invalid observer is set on a ColorObj.
    """

    def __init__(self, cobj: Any):
        super().__init__(cobj)
        self.message = "Invalid observer angle specified: %s" % cobj.observer
