from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeVar

import numpy
import numpy.typing
from numpy.linalg import pinv

from colormath import color_constants, color_objects

logger = logging.getLogger(__name__)


def _get_adaptation_matrix(
    wp_src: color_constants.ILLUMINANTS_TYPE | Sequence[float],
    wp_dst: color_constants.ILLUMINANTS_TYPE | Sequence[float],
    observer: color_constants.OBSERVERS_TYPE,
    adaptation: color_constants.CHROMATIC_ADAPTIONS,
):
    """
    Calculate the correct transformation matrix based on origin and target
    illuminants. The observer angle must be the same between illuminants.

    See colormath.color_constants.ADAPTATION_MATRICES for a list of possible
    adaptations.

    Detailed conversion documentation is available at:
    http://brucelindbloom.com/Eqn_ChromAdapt.html
    """
    # Get the appropriate transformation matrix, [MsubA].
    m_sharp = color_constants.ADAPTATION_MATRICES[adaptation]

    # In case the white-points are still input as strings
    # Get white-points for illuminant
    if isinstance(wp_src, str):
        wp_src = color_constants.ILLUMINANTS[observer][wp_src]

    if isinstance(wp_dst, str):
        wp_dst = color_constants.ILLUMINANTS[observer][wp_dst]

    # Sharpened cone responses ~ rho gamma beta ~ sharpened r g b
    rgb_src = numpy.dot(m_sharp, wp_src)
    rgb_dst = numpy.dot(m_sharp, wp_dst)

    # Ratio of whitepoint sharpened responses
    m_rat = numpy.diag(rgb_dst / rgb_src)

    # Final transformation matrix
    ret = numpy.dot(numpy.dot(pinv(m_sharp), m_rat), m_sharp)
    return ret  # noqa: RET504  # TODO: figure out why this is ending up as Any


# noinspection PyPep8Naming
def apply_chromatic_adaptation(
    val_x: float,
    val_y: float,
    val_z: float,
    orig_illum: color_constants.ILLUMINANTS_TYPE | Sequence[float],
    targ_illum: color_constants.ILLUMINANTS_TYPE | Sequence[float],
    observer: color_constants.OBSERVERS_TYPE = "2",
    adaptation: color_constants.CHROMATIC_ADAPTIONS = "bradford",
) -> tuple[float, float, float]:
    """
    Applies a chromatic adaptation matrix to convert XYZ values between
    illuminants. It is important to recognize that color transformation results
    in color errors, determined by how far the original illuminant is from the
    target illuminant. For example, D65 to A could result in very high maximum
    deviance.

    An informative article with estimate average Delta E values for each
    illuminant conversion may be found at:

    http://brucelindbloom.com/ChromAdaptEval.html
    """
    wp_src: Sequence[float] = (
        color_constants.ILLUMINANTS[observer][orig_illum] if isinstance(orig_illum, str) else orig_illum
    )
    wp_dst: Sequence[float] = (
        color_constants.ILLUMINANTS[observer][targ_illum] if isinstance(targ_illum, str) else targ_illum
    )

    logger.debug("  \\* Applying adaptation matrix: %s", adaptation)
    # Retrieve the appropriate transformation matrix from the constants.
    transform_matrix = _get_adaptation_matrix(wp_src, wp_dst, observer, adaptation)

    # Stuff the XYZ values into a NumPy matrix for conversion.
    XYZ_matrix = numpy.array((val_x, val_y, val_z))
    # Perform the adaptation via matrix multiplication.
    result_matrix = numpy.dot(transform_matrix, XYZ_matrix)

    # Return individual X, Y, and Z coordinates.
    return result_matrix[0], result_matrix[1], result_matrix[2]


ColorT = TypeVar("ColorT", bound=color_objects.XYZColor)


def apply_chromatic_adaptation_on_color(
    color: ColorT,
    targ_illum: color_constants.ILLUMINANTS_TYPE,
    adaptation: color_constants.CHROMATIC_ADAPTIONS = "bradford",
) -> ColorT:
    """
    Convenience function to apply an adaptation directly to a Color object.
    """
    xyz_x = color.xyz_x
    xyz_y = color.xyz_y
    xyz_z = color.xyz_z
    orig_illum = color.illuminant
    observer = color.observer

    # Return individual X, Y, and Z coordinates.
    color.xyz_x, color.xyz_y, color.xyz_z = apply_chromatic_adaptation(
        xyz_x,
        xyz_y,
        xyz_z,
        orig_illum,
        targ_illum,
        observer=observer,
        adaptation=adaptation,
    )
    color.set_illuminant(targ_illum)
    return color
