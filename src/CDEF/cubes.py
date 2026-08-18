import numpy as np
import CDEF
from build123d import Box, fillet
from .cdef_utils import build123d_to_mesh


def cube(fillet_radius, side_length):
    """
    Create a cube with optional rounded edges.

    Parameters
    ----------
    fillet_radius : float
        Radius of the edge fillet. If 0, a sharp cube is returned.
    side_length : float
        Edge length of the cube.

    Returns
    -------
    Solid
        build123d solid.
    """

    cube = Box(side_length, side_length, side_length)

    if fillet_radius > 0:
        max_r = cube.max_fillet(cube.edges())
        r = min(fillet_radius, 0.99 * max_r)

        while True:
            try:
                cube = fillet(cube.edges(), radius=r)
                break
            except ValueError:
                r *= 0.99

    return cube

def cube_curve(fillet_radius, side_length, N=30000):
    cube_shape = cube(fillet_radius, side_length)

    mesh = build123d_to_mesh(cube_shape)
    cloud = CDEF.stl_cloud(mesh, N, sequence="halton")
    unitcurve = CDEF.scattering_mono(cloud, selfcorrelation=True)

    unitscattering = {
        "unitcurve": unitcurve,
        "box": CDEF.box_from_mesh(mesh),
        "volume": CDEF.volume_from_mesh(mesh),
        "N": N,
        "side_length": side_length,
        "fillet_radius": fillet_radius,
        "model": "cube",
    }

    unitscattering["filling_factor"] = (
        unitscattering["volume"]
        / np.prod(unitscattering["box"])
    )

    return unitscattering