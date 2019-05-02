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
#include "basic.h"     /* basic types and time measurement */
#include "inversion.h" /* factorizing a matrix */
#include "matrix.h"    /* matrix functions */
#include "miniblas.h"  /* our simple BLAS version */

/* ------------------------------------------------------------
 * Setup test matrices
 * ------------------------------------------------------------ */

pmatrix build_ddom_random_matrix(int n) { return new_diagrandom_matrix(n); }

/* ============================================================
 * Main program
 * ============================================================ */

int main(void) {
  pmatrix a, a_entry, a_block, b, I;
  double *aa, *aa_entry, *aa_block, *ab, *aI;
  pstopwatch sw = new_stopwatch();
  double err_abs, t;
  int n, blocks, resolution, minblocksize;

  printf("Testing matrix inversion\n");

  n = 8;         // Problem size
  blocks = 2;      // Number of subdivisions of rows and columns in each
                   // level of the recursion
  resolution = 4;  // Stop the recursion if current blocks are smaller
                   // than the resolution

  printf("============================================================\n");

  // Setup test matrix
  printf("Setting up test matrix of size %d x %d:\n", n, n);
  start_stopwatch(sw);
  a = new_diagrandom_matrix(n);
  aa = a->a;
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);
  printf("  %.3f KB\n", n * n * sizeof(double) / 1024.0);

  printf("============================================================\n");

  printf("Clone matrix and set up product and reference matrix:\n");
  start_stopwatch(sw);
  a_entry = clone_matrix(a);
  aa_entry = a_entry->a;
  a_block = clone_matrix(a);
  aa_block = a_block->a;
  b = new_zero_matrix(n, n);
  ab = b->a;
  I = new_identity_matrix(n);
  aI = I->a;
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Compute inverse, per-entry version:\n");
  start_stopwatch(sw);
  // Invert A
  invert(a_entry);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Evaluate inverse:\n");
  start_stopwatch(sw);
  // Evaluate A^{-1} <-- A^{-1}A
  gemm(false, false, n, n, n, 1.0, aa, n, aa_entry, n, 1.0, ab, n);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Compute error of inversion:\n");
  start_stopwatch(sw);
  // Compute error between I and A^{-1}A in Frobenius norm
  axpy(n * n, -1.0, aI, 1, ab, 1);
  err_abs = nrm2(n * n, ab, 1);
  printf("  Absolute max error: %.3e\n", err_abs);
  printf("  Relative max error: %.3e\n", err_abs / sqrt(1.0 * n));
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  clear_matrix(b);

  printf("\n\n");

  printf("============================================================\n");

  printf("Compute inverse, block version:\n");

  start_stopwatch(sw);
  // Decompose Invert A
  minblocksize = invert_block(a_block, blocks, resolution);
  t = stop_stopwatch(sw);
  printf("Used %d x %d blocks --> blocksize = %.3f KB\n", minblocksize,
         minblocksize, minblocksize * minblocksize * sizeof(double) / 1024.0);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Evaluate inverse:\n");
  start_stopwatch(sw);
  // Evaluate A^{-1} <-- A^{-1}A
  gemm(false, false, n, n, n, 1.0, aa, n, aa_block, n, 1.0, ab, n);
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("============================================================\n");

  printf("Compute error of inversion:\n");
  start_stopwatch(sw);
  // Compute error between I and A^{-1}A
  axpy(n * n, -1.0, aI, 1, ab, 1);
  err_abs = nrm2(n * n, ab, 1);
  printf("  Absolute max error: %.3e\n", err_abs);
  printf("  Relative max error: %.3e\n", err_abs / sqrt(1.0 * n));
  t = stop_stopwatch(sw);
  printf("  %.3f ms\n", t * 1.0e3);

  printf("\n\n");

  printf("Cleanup:\n");

  del_matrix(a);
  del_matrix(a_entry);
  del_matrix(a_block);
  del_matrix(b);
  del_matrix(I);

  return EXIT_SUCCESS;
}
