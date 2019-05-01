/*---------------------------------------------------------------*/
/*		     Numerische Programmierung  	 	 */
/* 	Serie 1 - LDL^T-Zerlegung mit BLAS	 	 */
/* ------------------------------------------------------------- */
/*	Autoren: Sven Christophersen				 */
/*---------------------------------------------------------------*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Additional libraries */
#include "basic.h"    /* basic types and time measurement */
#include "matrix.h"   /* matrix functions */
#include "miniblas.h" /* our simple BLAS version */

/* ------------------------------------------------------------
 * Setup test matrices
 * ------------------------------------------------------------ */

/* Simple 2 times 2 version */
pmatrix new_2x2_matrix() {
  pmatrix a;
  double *aa;

  a = new_matrix(2, 2);
  aa = a->a;

  aa[0] = -18.0;
  aa[1] = -15.0;
  aa[2] = -15.0;
  aa[3] = 3.0;

  return a;
}

pmatrix new_3x3_matrix() {
  pmatrix a;
  double *aa;

  a = new_matrix(3, 3);
  aa = a->a;


  aa[0] = 4.0;
  aa[1] = 12.0;
  aa[2] = -16.0;

  aa[3] = 12.0;
  aa[4] = 37.0;
  aa[5] = -43.0;

  aa[6] = -16.0;
  aa[7] = -43.0;
  aa[8] = 98.0;

  return a;
}

/* Simple 4 times 4 version */
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

pmatrix build_ddom_hilbert_matrix() {
  int n = 1024;

  return new_diaghilbert_matrix(n);
}

/* ------------------------------------------------------------
 * LDL^T decomposition using BLAS
 * ------------------------------------------------------------ */

static void eval_l(const pmatrix a, pvector x) {
  int n = a->rows;
  int lda = a->ld;
  double *aa = a->a;
  double *xx = x->x;
  int k;

  // aa[ 1 + 1 * lda]

  assert(a->cols == n);
  assert(x->rows == n);

  // TODO
  // this calculates the dot product of the vector starting at the pointer
  // represented by the inpiut pvector x and an offset and by a vector inside
  // the matrix. the algorithem anssures that, since we are working with an
  // inplace symetric LDL^T decomposition, that the diagonal of the matrix A is
  // treaded as 1's. Further more the algorithem needs to start from the higher
  // to the lower entries of the vector x since we override the values in x and
  // thus can only calculate the rest of the equation if we are sure we dont
  // need the values at the previous indices anymore.
  for (int i = n - 1; i > 0; i--) {
    xx[i] += dot(i, aa + i, lda, xx, 1);
  }
}

static void eval_d(const pmatrix a, pvector x) {
  int n = a->rows;
  int lda = a->ld;
  double *aa = a->a;
  double *xx = x->x;
  int k;

  assert(a->cols == n);
  assert(x->rows == n);

  // TODO
  for (int i = 0; i < n; i++) {
    xx[i] = aa[i + i * lda] * xx[i];
  }
}

static void eval_lt(const pmatrix a, pvector x) {
  int n = a->rows;
  int lda = a->ld;
  double *aa = a->a;
  double *xx = x->x;
  int k;

  assert(a->cols == n);
  assert(x->rows == n);

  // TODO
  for (int i = 0; i < n - 1; i++) {
    // this calculates the dot product of the vector starting at the pointer
    // represented by the inpiut pvector x and an offset and by a vector inside
    // the matrix. the algorithem assures that, since we are working with an
    // inplace symmetric LDL^T decomposition, that the diagonal of the matrix A
    // is treaded as 1's. So the pointers are offsettet by i+1 further more the
    // algorithem needs to start from the lower to the higher entries of the
    // vector x since we override the values saved in x and thus can only
    // calculate the rest of the equation if we are sure we dont need the values
    // at the previous indices anymore.
    xx[i] += dot(n - i - 1, aa + i + 1 + i * lda, 1, xx + i + 1, 1);
  }
}

static void eval_ldlt(const pmatrix a, pvector x) {
  // TODO

  // (L * D * L^T )x
  // (L * D * (L^T * x))
  // (L * (D * (L^T * x))

  eval_lt(a, x);
  eval_d(a, x);
  eval_l(a, x);
  
  // LDL^T 
}

static void decomp_ldlt(pmatrix a) {
  int n = a->rows;
  int lda = a->ld;
  double *aa = a->a;
  int k;

  assert(a->cols == n);

  // TODO

  // The loop iterates n times where n is the number of dimensions and decreases
  // the size of the matrix with each iteration

  // For this function we interpret the matrix given by a as
  // | A_00 A_1* |
  // | A_*1 A_** |

  for (int d = 0; d < n - 1; d++) {
    // Aussuming we interpret the pmatrix a as a symmetrical n x n matrix A,
    // where n is the number of rows then the scal function scales the vector
    // A_1* by a factor alpha. In this case aplha is 1 divided by the d'th
    // diagonal entry of A denoted by 1.0/aa[d + d * lda].
    scal(n - (d + 1), 1.0 / aa[d + d * lda], aa + d + 1 + d * lda, 1.0);
    

    // This is the symmetrical equivalent for the ger function which we can use
    // here since our matrix A is symmetrical. syr basically executes the
    // operation A ← αxy^T + A for a synmetric matrix
    syr('l',  // Since our values are stored in the lower (or left) part of the
              // matrix we pass the character 'l' as first parameter.
        n - (d + 1),  // the dimension of the vector x. this represents the
                      // dimension of A_*d
        -1.0 * aa[d + d * lda],  // the α value
        aa + (d + 1) + d * lda,  // the pointer to the start of the vector A_d*
        1,                       // incx for the A_*d vector x
        aa + (d + 1) + (d + 1) * lda,  // the pointer to  A_(d+1)(d+1) wich is
                                       // the first value of matrix A_**
        lda);                          // the leading dimension of lda
  }
}

/* ============================================================
 * Main program
 * ============================================================ */

int main(void) {
  pmatrix a;
  pvector b, x;
  double err_abs, norm_b;
  int n;
  int i;

  printf("Testing matrix LDL^T decomposition\n");

  // Setup test matrix
  // a = new_2x2_matrix();
  //  a = new_3x3_matrix();

  a = new_4x4_matrix();
  // a = build_ddom_hilbert_matrix(); // 1024 x 1024 matrix

  // Print the matrix values to stdout, for debugging purpose
  printf("Matrix before factorization:\n");
  print_matrix(a);

  n = a->rows;

  b = new_zero_vector(n);
  x = new_zero_vector(n);

  // Setup test vector
  for (i = 0; i < n; i++) {
    x->x[i] = 1.0 / (1.0 + i);
  }

  // Compute matrix-vector-product by standard gemv
  gemv(false, n, n, 1.0, a->a, a->ld, x->x, 1, b->x, 1);

  // Decompose A = L * D * L^T
  decomp_ldlt(a);
  // Print the matrix values to stdout, for debugging purpose
  printf("Matrix after factorization:\n");
  print_matrix(a);

  // Evaluate x <-- (L * D * L^T) * x
  eval_ldlt(a, x);

  // Compute error between b and the in-place product (L * D * L^T) * x
  norm_b = nrm2(n, b->x, 1);
  axpy(n, -1.0, b->x, 1, x->x, 1);
  err_abs = nrm2(n, x->x, 1);
  printf("  Absolute max error: %.3e\n", err_abs);
  printf("  Relative max error: %.3e\n", err_abs / norm_b);

  del_matrix(a);
  del_vector(x);
  del_vector(b);

  return EXIT_SUCCESS;
}
