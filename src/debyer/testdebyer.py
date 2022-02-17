import debyer
import numpy as np

# bipyramid = octahedron
octa_vertices = [ 
        (-1, 1, 0), (-1, -1, 0), (0, 0, -1),
        (1, -1, 0), (1, 1, 0), (0, 0, 1),
    ]
octa_triangles = [
        [0, 2, 1], [0, 4, 2], [0, 1, 5], [0, 5, 4],
        [3, 1, 2], [3, 5, 1], [3, 2, 4], [3, 4, 5],
    ]

# compute vertices

octa_mesh = np.array([[octa_vertices[vind] for vind in vx] for vx in octa_triangles], dtype=np.float64)

test=np.zeros((len(octa_triangles),3,3))

pt = debyer.makepoints(octa_mesh, 10)

print(pt)
#np.savetxt('octahedron_geom.dat', pt);

# create 50000 points

#ptlong = debyer.makepoints(octa_mesh, 50000)
print(debyer.debyer_ff(pt,0.1,1,0.1))


