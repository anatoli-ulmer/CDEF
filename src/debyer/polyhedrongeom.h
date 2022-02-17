#ifndef POLYHEDRONGEOM_H
#define POLYHEDRONGEOM_H
typedef double Point[3];
int winding_number(int nTriangles, double *coords, Point point);
void compute_bb(int nTriangles, double *coords, double *lower, double *upper);
#endif
