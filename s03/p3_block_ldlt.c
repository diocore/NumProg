/*---------------------------------------------------------------*/
/*		     Numerische Programmierung  	 	 */
/* 	Serie 3 - Block LDL^T-Zerlegung mit BLAS	 	 */
/* ------------------------------------------------------------- */
/*	Autoren: Sven Christophersen				 */
/*	Versionsnummer:	1					 */
/*---------------------------------------------------------------*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Additional libraries */
#include "basic.h"          /* basic types and time measurement */
#include "factorizations.h" /* factorizing a matrix */
#include "matrix.h"         /* matrix functions */
#include "miniblas.h"       /* our simple BLAS version */

/* ------------------------------------------------------------
 * Setup test matrices
 * ------------------------------------------------------------ */

pmatrix build_ddom_hilbert_matrix(int n) { return new_diaghilbert_matrix(n); }

/* ============================================================
 * Main program
 * ============================================================ */

pmatrix new_4x4_matrix() {
  pmatrix a;
  double *aa;

  a = new_matrix(4, 4);
  aa = a->a;

  aa[0] = 2.0;
  aa[1] = 4.0;
  aa[2] = 6.0;
  aa[3] = -2.0;

  aa[4] = 4.0;
  aa[5] = 10.0;
  aa[6] = 1.0;
  aa[7] = -5.0;

  aa[8] = 6.0;
  aa[9] = 1.0;
  aa[10] = -1.0;
  aa[11] = 4.0;

  aa[12] = -2.0;
  aa[13] = -5.0;
  aa[14] = 4.0;
  aa[15] = 1.0;

  return a;
}

int main(void) {
  pmatrix a, a_entry, a_block;
  pvector b, x_entry, x_block;
  pstopwatch sw = new_stopwatch();
  double err_abs, norm_b, t;
  int n, m;
  int i;

  printf("Testing matrix LDL^T decomposition\n");

  n = 4;  // Problem size
  m = 2;  // Number of subdivision of rows and columns

  printf("============================================================\n");

  // Setup test matrix
  printf("Setting up test matrix of size %d x %d:\n", n, n);
  start_stopwatch(sw);
  a = build_ddom_hilbert_matrix(n);
  a = new_4x4_matrix(4, 4);

  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);
  printf("  %.3f KB\n", n * n * sizeof(double) / 1024.0);

  printf("============================================================\n");

  printf("Clone matrix:\n");
  start_stopwatch(sw);
  a_entry = clone_matrix(a);
  a_block = clone_matrix(a);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Setting up test vector:\n");
  start_stopwatch(sw);
  b = new_zero_vector(n);
  x_entry = new_zero_vector(n);
  x_block = new_zero_vector(n);

  // Setup test vector
  for (i = 0; i < n; i++) {
    x_entry->x[i] = x_block->x[i] = 1.0 / (1.0 + i);
  }
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);
  printf("============================================================\n");

  printf("Computing right-hand-side vector by standard MVM:\n");
  start_stopwatch(sw);
  // Compute matrix-vector-product by standard gemv
  gemv(false, n, n, 1.0, a->a, a->ld, x_entry->x, 1, b->x, 1);
  norm_b = nrm2(n, b->x, 1);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("\n\n");

  printf("============================================================\n");

  printf("Compute LDL^T-factorization, per-entry version:\n");
  start_stopwatch(sw);
  // Decompose A = L * D * L^T
  ldltdecomp(a_entry);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Evaluate factorization:\n");
  start_stopwatch(sw);
  // Evaluate x <-- (L * D * L^T) * x
  eval_ldlt(a_entry, x_entry);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Compute error of factorization:\n");
  start_stopwatch(sw);
  // Compute error between b and the in-place product (L * D * L^T) * x
  axpy(n, -1.0, b->x, 1, x_entry->x, 1);
  err_abs = nrm2(n, x_entry->x, 1);
  printf("  Absolute max error: %.3e\n", err_abs);
  printf("  Relative max error: %.3e\n", err_abs / norm_b);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("\n\n");

  printf("============================================================\n");

  printf("Compute LDL^T-factorization, block version:\n");
  printf("Using %d x %d blocks --> blocksize = %.3f KB\n", n / m, n / m,
         n / m * n / m * sizeof(double) / 1024.0);
  start_stopwatch(sw);
  // Decompose A = L * D * L^T
  block_ldltdecomp(a_block, m);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Evaluate factorization:\n");
  start_stopwatch(sw);
  // Evaluate x <-- (L * D * L^T) * x
  eval_ldlt(a_block, x_block);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Compute error of factorization:\n");
  start_stopwatch(sw);
  // Compute error between b and the in-place product (L * D * L^T) * x
  axpy(n, -1.0, b->x, 1, x_block->x, 1);
  err_abs = nrm2(n, x_block->x, 1);
  printf("  Absolute max error: %.3e\n", err_abs);
  printf("  Relative max error: %.3e\n", err_abs / norm_b);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("\n\n");

  printf("Cleanup:\n");

  del_matrix(a);
  del_matrix(a_entry);
  del_matrix(a_block);
  del_vector(x_entry);
  del_vector(x_block);
  del_vector(b);

  return EXIT_SUCCESS;
}

// int main(void) {
//   pmatrix a;
//   pvector b, x;
//   double err_abs, norm_b;
//   int n;
//   int i;

//   printf("Testing matrix LDL^T decomposition\n");

//   // Setup test matrix
//   // a = new_2x2_matrix();
//   //  a = new_3x3_matrix();

//   a = new_4x4_matrix();
//   // a = build_ddom_hilbert_matrix(); // 1024 x 1024 matrix

//   // Print the matrix values to stdout, for debugging purpose
//   printf("Matrix before factorization:\n");
//   print_matrix(a);

//   n = a->rows;

//   b = new_zero_vector(n);
//   x = new_zero_vector(n);

//   // Setup test vector
//   for (i = 0; i < n; i++) {
//     x->x[i] = 1.0 / (1.0 + i);
//   }

//   // Compute matrix-vector-product by standard gemv
//   gemv(false, n, n, 1.0, a->a, a->ld, x->x, 1, b->x, 1);

//   // Decompose A = L * D * L^T
//   block_ldltdecomp(a, 2);
//   // Print the matrix values to stdout, for debugging purpose
//   printf("Matrix after factorization:\n");
//   print_matrix(a);

//   // Evaluate x <-- (L * D * L^T) * x
//   eval_ldlt(a, x);

//   // Compute error between b and the in-place product (L * D * L^T) * x
//   norm_b = nrm2(n, b->x, 1);
//   axpy(n, -1.0, b->x, 1, x->x, 1);
//   err_abs = nrm2(n, x->x, 1);
//   printf("  Absolute max error: %.3e\n", err_abs);
//   printf("  Relative max error: %.3e\n", err_abs / norm_b);

//   del_matrix(a);
//   del_vector(x);
//   del_vector(b);

//   return EXIT_SUCCESS;
// }