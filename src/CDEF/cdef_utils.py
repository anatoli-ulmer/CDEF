import numpy as np

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