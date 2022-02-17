/*  debyer -- program for calculation of diffration patterns
 *  Copyright 2006 Marcin Wojdyr
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  Contains data structures and functions used by debyer and other programs
 *  for doing computations of PDFs, diffraction patters, etc.
 */
#ifndef DEBYER_DEBYER_H_
#define DEBYER_DEBYER_H_

#include <math.h>
#include <time.h> /* time_t */

#if HAVE_CONFIG_H
#  include <config.h>
#else
#  define VERSION "0.4" /* keep the same as in configure.ac */
#endif

#ifdef USE_MPI
# include "mpi.h"
#endif

#ifdef USE_SINGLE
# define dbr_real float
# define DBR_F    "%f"
#else
# define dbr_real double
# define DBR_F    "%lf"
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef dbr_real dbr_xyz[3];
typedef union {
	char name[8];
	double weight;
} dbr_symbol;

typedef struct dbr_atom dbr_atom;
typedef struct dbr_atoms dbr_atoms;
typedef struct irdf irdf;
typedef struct irdfs irdfs;
typedef struct dbr_cell dbr_cell;
typedef struct dbr_cells dbr_cells;
typedef struct dbr_pbc dbr_pbc;
typedef struct dbr_pbc_prop dbr_pbc_prop;
typedef struct dbr_picker dbr_picker;


typedef enum OutputKind {
    output_xray,
    output_neutron,
	output_cont, /* sampled continuum, dbr_symbol stores excess densities */
	output_cont_noself, /* sampled continuum, dbr_symbol stores excess densities */
    output_sf, /*scattering factor, a.k.a. total scattering structure function*/
    output_rdf, /* RDF, R(r) */
    output_pdf,  /* PDF, g(r) */
    output_rpdf, /* reduced PDF, G(r) */
    output_none
} OutputKind;

/* Structure used for passing atom name and coordinates.
 * It can be null, see is_null() and nullify(). */
struct dbr_atom
{
    dbr_xyz xyz;
    dbr_symbol name;
};

/* for storing atoms of one type */
struct dbr_atoms
{
    dbr_symbol name; /* atom symbol */
    int count; /* number of atoms */
    int asize; /* allocated capacity of xyz */
    dbr_xyz *xyz; /* coordinates of atoms */
    int *indices; /* indices (IDs) of atoms from original file */
};

/* used in struct irdfs */
struct irdf
{
    dbr_symbol at1, at2; /* atom names */
    int c1, c2; /* number of atoms of at1 and at2 types */
    int sample; /* if nonzero, number of sampled atoms at1 */
    int* nn; /* pair correlation function */
};

/* used for storing ID, which is than used for
 * calculation of x-ray/neutron patterns and PDFs.
 */
struct irdfs
{
    dbr_real step;
    int rdf_bins; /* data[n].nn length */
    int pair_count; /* data length */
    int symbol_count; /* number of different atom types */
    dbr_symbol* atom_symbols; /* symbol (name) of atom of each type */
    int* atom_counts; /* number of atoms of each type */
    irdf* data;
    dbr_real density; /* auto-calculated numeric density, can be overwritten */
};

/* periodic boundary conditions, only parallelpiped PBC are supported
 */
struct dbr_pbc
{
    double v00, v01, v02, v10, v11, v12, v20, v21, v22;
};

/* properties of PBC (or other) box */
struct dbr_pbc_prop
{
    dbr_pbc vectors; /* box vectors */
    dbr_xyz lengths; /* lengths of box vectors */
    dbr_xyz cosines; /* cosines of box angles */
    dbr_xyz widths; /* perpendicular box widths */
    dbr_real volume; /* volume of box */
};

/* cell method is for finding neighbours
 */

struct dbr_cell
{
    int asize; /* allocated capacity of indices */
    int count; /* number of atoms in cell */
    int real; /* is cell real (1) or virtual, ie. mirror (0) */
    int original; /* -1 if cell is real, otherwise (virtual cell) index
                    of corresponding real cell */
    dbr_xyz *atoms;
    int *indices; /* indices (IDs) of atoms from original file */
    int neighbours[27]; /* indices of neighbour cells (includes itself) */
};

struct dbr_cells
{
    dbr_symbol name; /* atom symbol */
    int n[3]; /* number of "real" cells is n[0] * n[1] * n[2] */
    int v[3]; /* has PBC, ie. virtual cells (1) or not (0) */
    int a[3]; /* a_i = n_i + 2 * v[i] */
    int count; /* number of all cells = a[1] * a[2] * a[3] */
    dbr_pbc vectors; /* cell's size */
    dbr_cell *data; /* cell is accessed using data[get_cell_nr(...)] */
    int atom_count; /* sum of all atoms (in all real cells) */
};

/* it is used to sampling or to calculate rdf only from the part of system */
struct dbr_picker
{
    int all; /* 1/0, 1 means: no picking, all atoms are counted */
    dbr_real probab; /*in sampling mode - probability, in (0,1). Otherwise: 0. */
    int cut; /* 1/0, 1 means use min/max values */
    /* To set no limit, use +/- DBL_MAX */
    double x_min, x_max, y_min, y_max, z_min, z_max;
};

struct dbr_pdf_args
{
    OutputKind c;
    int include_partials;
    dbr_real pattern_from;
    dbr_real pattern_to;
    dbr_real pattern_step;
    dbr_real ro;
    char weight;
};

struct dbr_diffract_args
{
    OutputKind c;
    dbr_real pattern_from;
    dbr_real pattern_to;
    dbr_real pattern_step;
    dbr_real lambda;
    dbr_real ro;
    dbr_real cutoff;
    int sinc_damp;
	int do_weights;
};

dbr_real get_sq_dist(const dbr_real *xyz1, const dbr_real *xyz2);

#if defined(USE_MPI)
extern int dbr_nid; /* rank of process (0 if serial) */
#elif defined(_OPENMP)
extern int dbr_nid;
#pragma omp threadprivate(dbr_nid)
#else
static const int dbr_nid = 0;
#endif
extern int dbr_verbosity; /* 0 = normal, 1 = verbose */

void dbr_print_version();
void dbr_init(int *argc, char ***argv);
void dbr_finalize();
void dbr_abort(int err_code);
void dbr_mesg(const char *fmt, ...);
int dbr_get_elapsed();

int dbr_get_atoms(int n, dbr_atom* coords, dbr_atoms** result,
                  int store_indices);
int dbr_get_atoms_weight(int n, dbr_atom* coords, dbr_atoms** result,
                  int store_indices);
void free_dbr_atoms(dbr_atoms* xa);
irdfs calculate_irdfs(int n, dbr_atoms* xa, dbr_real rcut, dbr_real rquanta,
                      dbr_pbc pbc, const dbr_picker* picker,
                      const char* id_filename);
void free_irdfs(irdfs *rdfs);
void write_irdfs_to_file(irdfs rdfs, const char *filename);

int write_diffraction_to_file(struct dbr_diffract_args* dargs, irdfs rdfs,
                              const char *ofname);

int write_pdfkind_to_file(struct dbr_pdf_args* pdf_args, irdfs rdfs,
                          const char *ofname);

irdfs read_irdfs_from_file(const char *filename);

dbr_cells prepare_cells(dbr_pbc pbc, dbr_real rcut, dbr_atoms* xa);

dbr_cells* prepare_cells_all(dbr_pbc pbc, dbr_real rcut, dbr_atoms* xa, int n);
void free_cells_all(dbr_cells *cells, int n);

int dbr_is_direct(OutputKind k);
int dbr_is_inverse(OutputKind k);

dbr_pbc_prop get_pbc_properties(dbr_pbc pbc);
dbr_real* get_pattern(const irdfs* rdfs, struct dbr_diffract_args* dargs);

/* small algebra utils */
/* both args are 3x3 matrix, the first is input and the second output */
void dbr_inverse_3x3_matrix(const dbr_pbc a, double b[3][3]);

int dbr_is_atom_in_sector(const dbr_real *xyz, const dbr_picker* picker);

static inline double dbr_len3(double a, double b, double c)
    { return sqrt(a*a + b*b + c*c); }

static inline double dbr_len3v(double *a)
    { return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]); }


#ifdef __cplusplus

inline bool is_null(dbr_atom const& atom) { return atom.name[0] == '\0'; }
inline void nullify(dbr_atom& atom) { atom.name[0] = '\0'; }

inline int dbr_cell_original(const dbr_cell *a)
  { return a->original >= 0 ? a->original : a->neighbours[13]; }

inline
void dbr_diff3(const dbr_real *a, const dbr_real *b, dbr_real *r)
{
    int i;
    for (i = 0; i < 3; ++i)
        r[i] = a[i] - b[i];
}

inline
dbr_real dbr_dot3(const dbr_real *a, const dbr_real *b)
    { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

inline
dbr_real dbr_get_angle(const dbr_real *xyz1,
                       const dbr_real *xyz2,
                       const dbr_real *xyz3)
{
    int i;
    double t;
    dbr_real a[3], b[3];
    for (i = 0; i < 3; ++i) {
        a[i] = xyz1[i] - xyz2[i];
        b[i] = xyz3[i] - xyz2[i];
    }
    t = dbr_dot3(a,b) / sqrt(dbr_dot3(a,a) * dbr_dot3(b,b));
    /* it happens (very rarely) that due to rounding errors |t| > 1 */
    return fabs(t) < 1 ? acos(t) : 0.;
}
#endif /* __cplusplus */


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* DEBYER_DEBYER_H_ */

