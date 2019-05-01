
#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
  int rows;
  int cols;
  int ld;

  double *a;
} matrix;

typedef matrix *pmatrix;

typedef struct
{
  int rows;

  double *x;
} vector;

typedef vector *pvector;

/* ------------------------------------------------------------
 * Constructors and destructors
 * ------------------------------------------------------------ */

pmatrix
new_matrix(int rows, int cols);

pmatrix
new_zero_matrix(int rows, int cols);

pmatrix
new_identity_matrix(int rows);

pmatrix
clone_matrix(pmatrix a);

pmatrix
new_sub_matrix(pmatrix src, int rows, int roff, int cols, int coff);

void del_matrix(pmatrix a);

void del_sub_matrix(pmatrix a);

pvector
new_vector(int rows);

pvector
new_zero_vector(int rows);

void del_vector(pvector x);

pvector
matrix_col(pmatrix a, int i);

/* ------------------------------------------------------------
 * Example matrix
 * ------------------------------------------------------------ */

pmatrix
new_diaghilbert_matrix(int rows);

pmatrix
new_hilbert_matrix(int rows);

/* ------------------------------------------------------------
 * Simple utility functions
 * ------------------------------------------------------------ */

void clear_matrix(pmatrix x);

void clear_vector(pvector x);

void print_matrix(pmatrix a);

void print_vector(pvector x);

#endif
