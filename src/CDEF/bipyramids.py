import numpy as np
import os
from pathlib import Path
import yaml
import time
import pickle
from datetime import datetime

import CDEF
import numpy as np
import scipy.optimize
from scipy.optimize import root, Bounds, differential_evolution
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, NullFormatter

from build123d import Solid, Shell, Face, Wire, export_stl
from build123d import BuildPart, Circle, RegularPolygon, BuildSketch, offset, loft, Plane, chamfer, mirror, fillet, Axis



def bipyramid(fillet_radius, R, h, base_corners=5):
    
    def bipyramid_vertices(R, h, base_corners=5, center=(0,0,0), rotation=0.0):
        """Generate vertices of a bipyramid with a variable amount of base corners.

        Args:
            R (float): radius of the base.
            h (float): height of the apices above and below the base plane.
            center (tuple): (x, y, z) coordinates of the bipyramid center.
            rotation (float): rotation angle in radians applied to the base pentagon.

        Returns:
            List of 7 tuples [(x,y,z), ...] representing the bipyramid vertices.
        """
        cx, cy, cz = center
        vertices = []
        
        # 5 vertices of pentagonal base in XY plane with rotation
        for i in range(base_corners):
            angle = 2 * np.pi * i / base_corners + rotation
            x = cx + R * np.cos(angle)
            y = cy + R * np.sin(angle)
            z = cz
            vertices.append((x, y, z))
        
        # apex vertex at +h and -h
        vertices.append((cx, cy, cz + h))
        vertices.append((cx, cy, cz - h))
        
        return vertices

    center=(0,0,0)
    rotation=0.0
    vertices = bipyramid_vertices(R=R, h=h, base_corners=base_corners, center=center, rotation=rotation)

    # Create a convex hull from the vertices
    hull = ConvexHull(vertices).simplices.tolist()

    # Create faces from the vertex indices
    faces = []
    for face_vertex_indices in hull:
        corner_vertices = [vertices[i] for i in face_vertex_indices]
        faces.append(Face(Wire.make_polygon(corner_vertices)))

    # Create the solid from the Faces
    bipy = Solid(Shell(faces)).clean()
    if fillet_radius > 0:
        max_r = bipy.max_fillet(bipy.edges())
        r = np.min([fillet_radius, 0.99 * max_r])
        while True:
            try:
                bipy = fillet(bipy.edges(), radius=r)
                break
            except ValueError:
                r *= 1.001
    return bipy


def bipyramid2(fillet_radius, R, h, base_corners=5):
    
    with BuildPart() as bipy:
        with BuildSketch(Plane.XY.offset(h)) as bipy_top:
            Circle(R*1e-3)
        with BuildSketch() as bipy_base:
            RegularPolygon(radius=R, side_count=base_corners)
            offset(amount=fillet_radius)
        loft()
        with BuildSketch(Plane.XY.offset(-h)) as bipy_bot:
            Circle(R*1e-3)
        mirror(about=Plane.XY)

    return bipy
    

def build123d_to_mesh(shape, tolerance=1e-3):
    all_tris = []
    for f in shape.faces():
        tess = f.tessellate(tolerance)
        # Case 1: tess returns (vertices, triangles)
        if isinstance(tess, tuple) and len(tess) == 2:
            vertices, triangles = tess
            # Convert each vertex to (x,y,z)
            coords = [ (v.X, v.Y, v.Z) if hasattr(v, "X") else tuple(v) for v in vertices ]
            for tri in triangles:
                all_tris.append([ coords[i] for i in tri ])
        # Case 2: tess returns list of triangles directly
        else:
            for tri in tess:
                # Convert each Vector to tuple if needed
                all_tris.append([ (v.X, v.Y, v.Z) if hasattr(v, "X") else tuple(v) for v in tri ])
    return np.array(all_tris, dtype=np.float64)


def bipy_curve(fillet_radius, radius, height, base_corners=5, N=30000, model='bipyramid2'):
    # radius = max(radius, 1e-8)
    # height = max(height, 1e-8)
    # print(f"fillet_radius = {fillet_radius}, radius = {radius}, height = {height}, base_corners = {base_corners}")
    if model == 'bipyramid':
        bipy = bipyramid(fillet_radius=fillet_radius, R=radius, h=height, base_corners=base_corners)
    elif model == 'bipyramid2':
        bipy = bipyramid2(fillet_radius=fillet_radius, R=radius, h=height, base_corners=base_corners)
    mesh = build123d_to_mesh(bipy)
    cloud = CDEF.stl_cloud(mesh, N, sequence='halton')
    unitcurve = CDEF.scattering_mono(cloud, selfcorrelation=True)
    unitscattering = {}
    unitscattering['unitcurve'] = unitcurve
    # subtraction moved to CDEF_main.py
    # unitscattering['unitcurve'][:,1] -= 1/N
    # unitscattering['unitcurve'][:,1] -= np.min([np.min(unitscattering['unitcurve'][:,1])*.99, 1/N])
    unitscattering['box'] = CDEF.box_from_mesh(mesh)
    unitscattering['volume'] = CDEF.volume_from_mesh(mesh)
    unitscattering['filling_factor'] = unitscattering['volume'] / ( unitscattering['box'][0] * unitscattering['box'][1] * unitscattering['box'][2] )
    unitscattering['N'] = N
    unitscattering['radius'] = radius
    unitscattering['height'] = height
    unitscattering['fillet_radius'] = fillet_radius
    unitscattering['base_corners'] = base_corners
    unitscattering['model'] = model

    return unitscattering


def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def import_config(yaml_file):
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        raise FileNotFoundError(f"{yaml_file} not found")

    with open(yaml_path, "r") as f:
        sample_dir = yaml.safe_load(f)

    if sample_dir is None:
        raise ValueError(f"{yaml_file} is empty or malformed")

    # Convert bounds to tuples (for scipy differential_evolution)
    if "bounds" in sample_dir and sample_dir["bounds"] is not None:
        sample_dir["bounds"] = [tuple(b) for b in sample_dir["bounds"]]

    print("Loaded parameters:", sample_dir)
    
    return sample_dir
