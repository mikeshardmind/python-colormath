"""
Contains lookup tables, constants, and things that are generally static
and useful throughout the library.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, TypedDict

import numpy
import numpy.typing

# Used in Lab and Luv calculations, see https://www.color.org/icc1v42.pdf
CIE_E = 216.0 / 24389.0
CIE_K = 24389.0 / 27.0


OBSERVERS_TYPE = Literal["2", "10"]
ILLUMINANTS_2_DEGREE: TypeAlias = Literal["a", "b", "c", "d50", "d55", "d65", "d75", "e", "f2", "f7", "f11"]
ILLUMINANTS_10_DEGREE: TypeAlias = Literal["d50", "d55", "d65", "d75"]
ILLUMINANTS_TYPE: TypeAlias = ILLUMINANTS_2_DEGREE | ILLUMINANTS_10_DEGREE
ILLUMINANTS_MAP_TYPE = TypedDict(
    "ILLUMINANTS_MAP_TYPE",
    {
        "2": Mapping[ILLUMINANTS_TYPE, tuple[float, float, float]],
        "10": Mapping[ILLUMINANTS_TYPE, tuple[float, float, float]],
    },
)


# Observer Function and Illuminant Data
ILLUMINANTS: ILLUMINANTS_MAP_TYPE = {
    # 2 Degree Functions
    "2": {
        "a": (1.09850, 1.00000, 0.35585),
        "b": (0.99072, 1.00000, 0.85223),
        "c": (0.98074, 1.00000, 1.18232),
        "d50": (0.96422, 1.00000, 0.82521),
        "d55": (0.95682, 1.00000, 0.92149),
        "d65": (0.95047, 1.00000, 1.08883),
        "d75": (0.94972, 1.00000, 1.22638),
        "e": (1.00000, 1.00000, 1.00000),
        "f2": (0.99186, 1.00000, 0.67393),
        "f7": (0.95041, 1.00000, 1.08747),
        "f11": (1.00962, 1.00000, 0.64350),
    },
    # 10 Degree Functions
    "10": {
        "d50": (0.9672, 1.000, 0.8143),
        "d55": (0.958, 1.000, 0.9093),
        "d65": (0.9481, 1.000, 1.073),
        "d75": (0.94416, 1.000, 1.2064),
    },
}

OBSERVERS = ILLUMINANTS.keys()


# Chromatic Adaptation Matrices
# http://brucelindbloom.com/Eqn_ChromAdapt.html


CHROMATIC_ADAPTIONS: TypeAlias = Literal["xyz_scaling", "bradford", "von_kries"]

ADAPTATION_MATRICES: Mapping[CHROMATIC_ADAPTIONS, numpy.typing.NDArray[Any]] = {
    "xyz_scaling": numpy.array(
        (
            (1.00000, 0.00000, 0.00000),
            (0.00000, 1.00000, 0.00000),
            (0.00000, 0.00000, 1.00000),
        )
    ),
    "bradford": numpy.array(
        (
            (0.8951, 0.2664, -0.1614),
            (-0.7502, 1.7135, 0.0367),
            (0.0389, -0.0685, 1.0296),
        )
    ),
    "von_kries": numpy.array(
        (
            (0.40024, 0.70760, -0.08081),
            (-0.22630, 1.16532, 0.04570),
            (0.00000, 0.00000, 0.91822),
        )
    ),
}
