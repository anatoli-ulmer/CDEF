/*  debyer_wrap.c
 *
 *  (C) Copyright 2022 Physikalisch-Technische Bundesanstalt (PTB)
 *   Christian Gollwitzer
 *  
 *   This file is part of CDEF.
 *
 *   CDEF is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CDEF is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CDEF.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>

#ifdef DEBUG
#include <stdio.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "debyer.h"
#include "polyhedrongeom.h"

static inline double SQR(double x) { return x*x; }

double maxdist_bb(size_t Natoms, dbr_atom *atoms) {
    // compute maximum distance bounding box using OpenMP
    // - *much* faster 
    double minx =  atoms[0].xyz[0]; double maxx=minx;
    double miny =  atoms[0].xyz[1]; double maxy=miny;
    double minz =  atoms[0].xyz[2]; double maxz=minz;

#if defined(_MSC_VER) && !defined(__clang__)
	/* MSVC OpenMP support sucks - it lacks max reductions */
	int i;
#else
	size_t i; 
    #pragma omp parallel for reduction(max:maxx) \
        reduction(max:maxy) reduction(max:maxz) \
        reduction(min:minx) reduction(min:miny) reduction(min:minz)
#endif
    for (i=0; i<Natoms; i++) {
        double x=atoms[i].xyz[0];
        double y=atoms[i].xyz[1];
        double z=atoms[i].xyz[2];
        if (x<minx) { minx=x; }
        if (y<miny) { miny=y; }
        if (z<minz) { minz=z; }
        if (x>maxx) { maxx=x; }
        if (y>maxy) { maxy=x; }
        if (z>maxz) { maxz=x; }
    }

    return sqrt(SQR(maxx-minx)+SQR(maxy-miny)+SQR(maxz-minz));
}



/*  wrapped debyer calculation */
static PyObject* debyer_ff(PyObject* self, PyObject* args, PyObject * kwargs)
{
	static char* kwlist[] = {"points", "qfrom", "qto", "qstep", "selfcorrelation", "rbins", "cutoff", "sinc_damp", "zerobinstart", "zerobinend", NULL};

    PyArrayObject *coords;
    double from; double to; double step;
	int selfcorrelation = 1;
	int rbins = 1000;
	double cutoff = 0;
	double sinc_damp = 0;
	
	int zerobinstart = 0;
	int zerobinend   = -1;
    /*  parse single numpy array argument */
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ddd|piddii", kwlist, &PyArray_Type, &coords, &from, &to, &step, &selfcorrelation, &rbins, &cutoff, &sinc_damp, &zerobinstart, &zerobinend))
        return NULL;

    if (!PyArray_Check(coords) 
         && (PyArray_NDIM((PyArrayObject*)coords)==2)
         && PyArray_EquivTypenums(PyArray_TYPE(coords), NPY_FLOAT64)) {
            PyErr_SetString(PyExc_RuntimeError, "Coords must be a Nx3 or Nx4 array of coordinates in space with weights");
            return NULL;
    }

    const long Natoms_inp = PyArray_DIMS(coords)[0];
    const long dim =  PyArray_DIMS(coords)[1];

    if (dim != 3 && dim != 4) {
        PyErr_SetString(PyExc_RuntimeError, "Coords must be a Nx3 or Nx4 array of coordinates in space with optional weights");
        return NULL;
    }
    
    /* copy the data into an atom structure */
    dbr_atom* atoms = malloc(sizeof(dbr_atom)*Natoms_inp);

    long Natoms = 0;
    for (long i = 0; i < Natoms_inp; i++) {
        /* determine weight */
        if (dim == 4) {
            double weight = *((double*)(PyArray_GETPTR2(coords, i, 3)));
            if (weight == 0) continue;
            atoms[Natoms].name.weight = *((double*)(PyArray_GETPTR2(coords, i, 3)));
        } else {
            atoms[Natoms].name.weight = 1.0;
        }
        
        for (long j = 0; j < 3; ++j) {
            atoms[Natoms].xyz[j] = *((double*)(PyArray_GETPTR2(coords, i, j)));
        }
        Natoms++;
    }

    /* unit cell, just a single atom */
    dbr_pbc pbc = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    /* use all available atoms */
    dbr_picker picker;
    picker.all = 1;
    picker.cut = 0;
    picker.probab = 0.;

    dbr_atoms *xa = NULL;
    int tc;

    /* compute maximum distance */
    double rcut = maxdist_bb(Natoms, atoms)*2.0;

    /* group atoms by weight */
    tc = dbr_get_atoms_weight(Natoms, atoms, &xa, /*store_indices=*/0);
	
	double rquanta = rcut / rbins;
    irdfs rdfs = calculate_irdfs(tc, xa, rcut, rquanta, pbc, &picker, NULL);
    /* xa is free'd after this calculation */
#ifdef DEBUG 
	fprintf(stderr, "Length = %d\n", rdfs.rdf_bins );
	fprintf(stderr, "Pair Counts = %d\n", rdfs.pair_count );
	fprintf(stderr, "Step = %g\n", rdfs.step );
	fprintf(stderr, "Density = %g\n", rdfs.density );
	fprintf(stderr, "rcut = %g\n", rcut );
#endif
	/* Set initial histogram bins of each irdf to 0 */
	for (int pair = 0; pair < rdfs.pair_count; pair++) {
		for (int i=zerobinstart; i<zerobinend; i++) {
			rdfs.data[pair].nn[i] = 0;
		}
#ifdef DEBUG
		fprintf(stderr, "Zeroing initial bins from %d to %d\n", zerobinstart, zerobinend);
#endif

	}

    long Nbins =  (to - from) / step;

    npy_intp dims[2];
    dims[0] = Nbins;
    dims[1] = 2;
    PyArrayObject *ff =  (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    

    struct dbr_diffract_args dargs;
	if (selfcorrelation) {
		dargs.c = output_cont;
	} else {
		dargs.c = output_cont_noself;
	}
	
    dargs.pattern_from = from;
    dargs.pattern_to = to;
    dargs.pattern_step = step;
    dargs.lambda = 0.;
    dargs.ro = -1.;
    dargs.cutoff =  cutoff;
    dargs.sinc_damp = sinc_damp;
    dargs.do_weights = (dim == 4);

    dbr_real *result = get_pattern(&rdfs, &dargs);

    for (Py_ssize_t i = 0; i < Nbins; ++i) {
        *((double *)(PyArray_GETPTR2(ff, i, 0))) = from + (i+0.5)*step;
        *((double *)(PyArray_GETPTR2(ff, i, 1))) = result[i]/Natoms;

    }
    
    free(result);
    free_irdfs(&rdfs);
	free(atoms);

    //Py_INCREF(ff);
    return (PyObject*)ff;
}

/* bit twiddling */
static uint64_t bitreverse(uint64_t input, uint64_t length) {
    // compute bit reversal for input with length bits
    uint64_t result = 0;
    for (uint64_t bit = 0; bit < length; ++bit) {
        result <<= 1;
        result  |= input % 2;
        input  >>= 1;
    }
    return result;
}

static uint64_t nbits(uint64_t input) {
    // count number of bits in input
    uint64_t result = 0;
    while (input > 0) {
        ++result;
        input >>= 1;
    }
    return result;
}

/* Generator for scrambled Halton sequence according to
 * Kocis, L., & Whiten, W. J. (1997). 
 * Computational investigations of low-discrepancy sequences. 
 * ACM Transactions on Mathematical Software (TOMS), 23(2), 266-294.
 */
static struct {
    uint64_t index;
    uint64_t *primes;
    uint64_t **KW_Perm;
    size_t   Nprimes;
    uint64_t maxprime;
} halton_state;

static void init_primes();
static void init_kocis_whiten();

static void init_halton() {
    halton_state.index = 1;
    init_primes();
    init_kocis_whiten();
}

static void init_primes() {
    //fprintf(stderr, "Initialize primes\n");
    halton_state.Nprimes = 50;
    halton_state.primes=malloc(sizeof(*(halton_state.primes))*halton_state.Nprimes);
    size_t i = 0;
    halton_state.primes[i++]=2; 
    uint64_t candidate = 2;
    while (i < halton_state.Nprimes) {
        int prime = 1;
        do {
            ++candidate;
            prime = 1;
            for (size_t j = 0; j < i; ++j) {
               // fprintf(stderr, "Testing %d / %d\n", (int)candidate, (int)halton_state.primes[j]);
                if (candidate % halton_state.primes[j] == 0) {
                    prime = 0;
                    break;
                }
            }
        } while (!prime);
       // fprintf(stderr, "%d\n", (int)candidate);
        halton_state.primes[i++]=candidate;
    }
    halton_state.maxprime = candidate;
}

void init_kocis_whiten() {
    const uint64_t maxbits = nbits(halton_state.maxprime);
    const uint64_t maxi = ((uint64_t)1) << maxbits;
    
    // initialize 2D array
    halton_state.KW_Perm = malloc(sizeof(void*)*halton_state.Nprimes);
    uint64_t *kwlengths = malloc(sizeof(uint64_t)*halton_state.Nprimes);

    for (Py_ssize_t i=0; i < halton_state.Nprimes; i++) {
        halton_state.KW_Perm[i] = malloc(sizeof(**halton_state.KW_Perm)*halton_state.maxprime);
        kwlengths[i]=0;
    }

    
    for (uint64_t i = 0; i < maxi; ++i) {
        const uint64_t reversed = bitreverse(i, maxbits);
        for (size_t j = 0; j < halton_state.Nprimes; ++j) {
            if (reversed < halton_state.primes[j]) {
                uint64_t ind = kwlengths[j];
                halton_state.KW_Perm[j][ind]=reversed;
                kwlengths[j]++;
            }
        }
    }
    free(kwlengths);
}

static PyObject* halton_reset(PyObject* self, PyObject* args) {
    halton_state.index = 0;
    Py_RETURN_NONE;
}

static double halton(uint64_t index, uint64_t dim) {
    if (dim >= halton_state.Nprimes) { 
        /* error("Not enough primes."); */
        return nan("");
    }
    const uint64_t base = halton_state.primes[dim];
    const double inv_base = 1.0 / ((double) base);
    double result = 0, reverse_base = inv_base;
    /// \bug definition of seeded_index can induce sequence overlap
    while (index > 0) {
        const uint64_t  digit = (uint64_t) (index % base);
        // scrambling according to Kocis & Whiten
        const uint64_t sdigit = halton_state.KW_Perm[dim][digit];
        result += ((double) sdigit) * reverse_base;
        reverse_base *= inv_base,
        index /= base;
    }
    return result;
}

/*  Halton sequence generator */
static PyObject* kwhalton_wrap(PyObject* self, PyObject* args)
{
   
    Py_ssize_t Npoints, Ndim;
    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "nn", &Npoints, &Ndim)) {
        return NULL;

   }

    npy_intp dims[2];
    dims[0] = Npoints;
    dims[1] = Ndim;
    PyArrayObject *result =  (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    
    for (Py_ssize_t i = 0; i < Npoints; ++i) {
        for (Py_ssize_t j = 0; j < Ndim; ++j) {
            *((double *)(PyArray_GETPTR2(result, i, j))) = halton(halton_state.index, j);
        }
        halton_state.index++;

    }


    Py_INCREF(result);
    return (PyObject*)result;
}


/*  wrapped point generator for triangle mesh */
static PyObject* makepoints(PyObject* self, PyObject* args)
{

    PyArrayObject *trianglemesh = NULL;
    PyArrayObject *gridpoints = NULL;
	int Npoints = -1;

	PyArrayObject *result = NULL;

	/* Two ways to invoke:
	 *  + Mesh array + single integer for number of Halton points
	 *  + Mesh array + grid point array for preset points
    */

    if (PyArg_ParseTuple(args, "O!i", &PyArray_Type, &trianglemesh, &Npoints)) {

	} else {

		if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &trianglemesh, &PyArray_Type, &gridpoints))
			return NULL;
		
		PyErr_Clear();
	}

    if (!PyArray_Check(trianglemesh) 
         && PyArray_NDIM((PyArrayObject*)trianglemesh)==3 
         && PyArray_EquivTypenums(PyArray_TYPE(trianglemesh), NPY_FLOAT64)) {
            PyErr_SetString(PyExc_RuntimeError, "Trianglemesh must be a Nx3x3 array of doubles, corresponding to the coordinates of the vertices");
            return NULL;
    }

	/* check for point grid */
	if (!gridpoints) {
		/* grid points not explicitly given - check for positive number of Halton points */
		if (Npoints <= 0) {
            PyErr_SetString(PyExc_RuntimeError, "Number of Halton points must be positive");
            return NULL;
		}
	} else {
		if (!PyArray_Check(gridpoints) 
			 && PyArray_NDIM((PyArrayObject*)gridpoints)==2 
			 && PyArray_EquivTypenums(PyArray_TYPE(gridpoints), NPY_FLOAT64)) {
				PyErr_SetString(PyExc_RuntimeError, "Grid points must be a Nx3 array of doubles, corresponding to the coordinates of the point cloud");
				return NULL;
		}

		Npoints = PyArray_DIMS(gridpoints)[0];
		const long griddim = PyArray_DIMS(gridpoints)[1];

		if (griddim != 3) {
				PyErr_SetString(PyExc_RuntimeError, "Grid points must be a Nx3 array of doubles, corresponding to the coordinates of the point cloud");
				return NULL;
		}
	}

    /* Check for valid mesh and copy coordinates */
	const long nTriangles = PyArray_DIMS(trianglemesh)[0];
    const long nvert =  PyArray_DIMS(trianglemesh)[1];
	const long ndim  = PyArray_DIMS(trianglemesh)[2];

    if (nvert != 3 || ndim != 3) {
        PyErr_SetString(PyExc_RuntimeError, "Trianglemesh must be a Nx3x3 array doubles");
        return NULL;
    }

    if (Npoints < 1) {
        PyErr_SetString(PyExc_RuntimeError, "Nunmber of points must be at least 1");
        return NULL;
    }
    
    
	double *coords = malloc(sizeof(double)*nTriangles*9);

	for (long ind = 0; ind < nTriangles; ind++) {
		double *base = &coords[9*ind];
		for (int v=0; v < 3; v++) {
			for (int d=0; d < 3; d++) {
				base[v*3 + d] = *((double*)(PyArray_GETPTR3(trianglemesh, ind, v, d)));
			}
		}
	}
	

	Point lower, upper;
	compute_bb(nTriangles, coords, lower, upper);
#ifdef DEBUG
	fprintf(stderr, "Bounding Box: (%g  -  %g)  (%g  -  %g)  (%g  -  %g)\n", lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]); 
#endif
	
    npy_intp dims[2];
    dims[0] = Npoints;
    dims[1] = 3;
    
	result =  (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    
	int ind = 0;
    for (Py_ssize_t try = 0; try < Npoints; ++try) {		
		Point p;
		
		if (gridpoints) {
			// retrieve the coordinates from the numpy array
			for (int d=0; d < 3; d++) {
				double qrnd = *((double*)(PyArray_GETPTR2(gridpoints, try, d)));
				p[d] = lower[d] + qrnd*(upper[d] - lower[d]);
			}
			
			//printf("Testing (%g, %g, %g)\n", p[0],p[1],p[2]);

		} else {
			// draw a Halton point
			for (int d = 0; d < 3; d++) {
				double qrnd = halton(halton_state.index, d);
				p[d] = lower[d] + qrnd*(upper[d] - lower[d]);
			}

			halton_state.index++;
		}

		if (winding_number(nTriangles, coords, p) != 0) {
			for (int d = 0; d < 3; d++) {
				*((double *)(PyArray_GETPTR2(result, ind, d))) = p[d];
			}
			ind++;
		}
    }
#ifdef DEBUG	
	printf("Filling fraction: %g\n", ((double)ind)/Npoints);
#endif
	// Resize the output array to cut off the superfluous elements
	npy_intp out_dimptr[2];
	out_dimptr[0] = ind;
	out_dimptr[1] = 3;

	PyArray_Dims out_dims;
    out_dims.ptr = out_dimptr;
	out_dims.len = 2;


	if (!PyArray_Resize(result, &out_dims, false, NPY_CORDER)) {
		printf("Resizing has failed\n");
		result = NULL;
	}

	free(coords);
	
	return (PyObject*)result;
}

/*  wrapped point generator for triangle mesh */
static PyObject* findinside(PyObject* self, PyObject* args)
{

    PyArrayObject *trianglemesh = NULL;
    PyArrayObject *gridpoints = NULL;
	

	/* Two ways to invoke:
	 *  Mesh array + grid point array
    */

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &trianglemesh, &PyArray_Type, &gridpoints))
		return NULL;
		
    if (!PyArray_Check(trianglemesh) 
         && PyArray_NDIM((PyArrayObject*)trianglemesh)==3 
         && PyArray_EquivTypenums(PyArray_TYPE(trianglemesh), NPY_FLOAT64)) {
            PyErr_SetString(PyExc_RuntimeError, "Trianglemesh must be a Nx3x3 array of doubles, corresponding to the coordinates of the vertices");
            return NULL;
    }

	if (!PyArray_Check(gridpoints) 
		 && PyArray_NDIM((PyArrayObject*)gridpoints)==2 
		 && PyArray_EquivTypenums(PyArray_TYPE(gridpoints), NPY_FLOAT64)) {
			PyErr_SetString(PyExc_RuntimeError, "Grid points must be a Nx3 array of doubles, corresponding to the coordinates of the point cloud");
			return NULL;
	}
	
	const int Npoints = PyArray_DIMS(gridpoints)[0];
	const long griddim = PyArray_DIMS(gridpoints)[1];

	if (griddim != 3) {
			PyErr_SetString(PyExc_RuntimeError, "Grid points must be a Nx3 array of doubles, corresponding to the coordinates of the point cloud");
			return NULL;
	}

    /* Check for valid mesh and copy coordinates */
	const long nTriangles = PyArray_DIMS(trianglemesh)[0];
	const long nvert =  PyArray_DIMS(trianglemesh)[1];
	const long ndim  = PyArray_DIMS(trianglemesh)[2];

    if (nvert != 3 || ndim != 3) {
        PyErr_SetString(PyExc_RuntimeError, "Trianglemesh must be a Nx3x3 array doubles");
        return NULL;
    }

    if (Npoints < 1) {
        PyErr_SetString(PyExc_RuntimeError, "Nunmber of points must be at least 1");
        return NULL;
    }
    
    
	double *coords = malloc(sizeof(double)*nTriangles*9);

	for (long ind = 0; ind < nTriangles; ind++) {
		double *base = &coords[9*ind];
		for (int v=0; v < 3; v++) {
			for (int d=0; d < 3; d++) {
				base[v*3 + d] = *((double*)(PyArray_GETPTR3(trianglemesh, ind, v, d)));
			}
		}
	}
	

    npy_intp dims[1];
	dims[0] = Npoints;
	PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_BOOL);
    
    for (Py_ssize_t try = 0; try < Npoints; ++try) {		
		Point p;
	
		// retrieve the coordinates from the numpy array
		for (int d=0; d < 3; d++) {
			p[d] = *((double*)(PyArray_GETPTR2(gridpoints, try, d)));
		}
		
		*((char *)(PyArray_GETPTR1(result, try))) = (winding_number(nTriangles, coords, p) != 0) ;
    }
 	
	free(coords);
	
	return (PyObject *)result;
}


/*  compute filling fraction by MC integration */
static PyObject* filling_fraction(PyObject* self, PyObject* args)
{

    PyArrayObject *trianglemesh;
	int Npoints;

    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &trianglemesh, &Npoints))
        return NULL;

    if (!PyArray_Check(trianglemesh) 
         && PyArray_NDIM((PyArrayObject*)trianglemesh)==3 
         && PyArray_EquivTypenums(PyArray_TYPE(trianglemesh), NPY_FLOAT64)) {
            PyErr_SetString(PyExc_RuntimeError, "Trianglemesh must be a Nx3x3 array of doubles, corresponding to the coordinates of the vertices");
            return NULL;
    }

	const long nTriangles = PyArray_DIMS(trianglemesh)[0];
	const long nvert =  PyArray_DIMS(trianglemesh)[1];
	const long ndim  = PyArray_DIMS(trianglemesh)[2];

    if (nvert != 3 || ndim != 3) {
        PyErr_SetString(PyExc_RuntimeError, "Trianglemesh must be a Nx3x3 array doubles");
        return NULL;
    }

    if (Npoints < 1) {
        PyErr_SetString(PyExc_RuntimeError, "Number of points must be at least 1");
        return NULL;
    }
    
    
	double *coords = malloc(sizeof(double)*nTriangles*9);

	for (long ind = 0; ind < nTriangles; ind++) {
		double *base = &coords[9*ind];
		for (int v=0; v < 3; v++) {
			for (int d=0; d < 3; d++) {
				base[v*3 + d] = *((double*)(PyArray_GETPTR3(trianglemesh, ind, v, d)));
			}
		}
	}
	
	Point lower, upper;
	compute_bb(nTriangles, coords, lower, upper);
#ifdef DEBUG	
	fprintf(stderr, "Bounding Box: (%g  -  %g)  (%g  -  %g)  (%g  -  %g)\n", lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]); 
#endif
	int ind = 0;
    for (Py_ssize_t try = 0; try < Npoints && ind < Npoints; ++try) {
		
		Point p;

		for (int d = 0; d < 3; d++) {
			double qrnd = halton(halton_state.index, d);
			p[d] = lower[d] + qrnd*(upper[d] - lower[d]);
        }

		halton_state.index++;

		if (winding_number(nTriangles, coords, p) != 0) {
			ind++;
		}
    }
    
	free(coords);
    
	printf("Number of hits: %d points: %d\n", ind, Npoints); 

    return PyFloat_FromDouble((double)(ind) / (double)(Npoints));
}


/* structure of a binary STL file */
#pragma pack(push, 1)
typedef struct {
	uint8_t junk[80];
	uint32_t nTriangles;
} STL_HEADER;

typedef struct {
	float normal[3];
	float V[3][3];
	uint16_t attrib;
} STL_TRI;
#pragma pack(pop)


/*  Read a binary STL in suitable format */
static PyObject* read_stl(PyObject* self, PyObject* args)
{
	char *fname;
	/*  parse single numpy array argument */
	if (!PyArg_ParseTuple(args, "s", &fname)) {
		return NULL;
	}

	FILE* stl=fopen(fname, "rb");
	if (!stl) {
		PyErr_SetString(PyExc_RuntimeError, "Can't open STL file");
        return NULL;
    }
    
	STL_HEADER head;
	fread(&head, sizeof(head), 1, stl);

#ifdef DEBUG
	printf("STL file has %d triangles \n", head.nTriangles);
	printf("Expected size of file: %ld bytes \n", (long) (sizeof(head)+head.nTriangles*sizeof(STL_TRI)));
#endif

	npy_intp dims[3];
	dims[0] = head.nTriangles;
	dims[1] = 3;
	dims[2] = 3;
	PyArrayObject *result =  (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_FLOAT64);

	STL_TRI *stl_raw = malloc(sizeof(STL_TRI) * head.nTriangles);
	/* read complete file into memory */
	fread(stl_raw, sizeof(STL_TRI), head.nTriangles, stl);
	fclose(stl);

	/* convert into numpy array */
	for (uint32_t ind=0; ind<head.nTriangles; ind++) {
		for (int v = 0; v < 3; v++) {
			for (int d = 0; d < 3; d++) {
				*((double *)(PyArray_GETPTR3(result, ind, v, d))) = stl_raw[ind].V[v][d];
			}
		}
	}

	free(stl_raw);

	Py_INCREF(result);
	return (PyObject *)result;
}


struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyMethodDef debyer_methods[] = {
    {"debyer_ff", debyer_ff, METH_VARARGS|METH_KEYWORDS, "compute a scattering function"},
    {"makepoints", makepoints, METH_VARARGS, "put quasirandom points into a triangle mesh"},
    {"filling_fraction", filling_fraction, METH_VARARGS, "compute the filling fraction of a mesh inside its bounding box"},
    {"findinside", findinside, METH_VARARGS, "test which points from a cloud are inside a triangle mesh"},
    {"kwhalton", kwhalton_wrap, METH_VARARGS, "compute quasi-random number (kocis-whiten scrambled Halton sequence)"},
    {"halton_reset", halton_reset, METH_NOARGS, "Reset halton sequence generator"},
    {"read_stl", read_stl, METH_VARARGS, "Read a binary STL in a mesh format suitable for makepoints"},
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int debyer_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int debyer_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "debyer",
        NULL,
        sizeof(struct module_state),
        debyer_methods,
        NULL,
        debyer_traverse,
        debyer_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_debyer(void)

#else
#define INITERROR return

void
initdebyer(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("debyer", debyer_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("debyer.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

     /* IMPORTANT: this must be called */
     import_array();
     dbr_init(NULL, NULL);
     init_halton();
     
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}


