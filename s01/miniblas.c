
#include "miniblas.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------
 * Level 1
 * ------------------------------------------------------------ */

/* Scale a vector */
void scal(int n, real alpha, real *x, int incx) {
  int i;

  for (i = 0; i < n; i++) x[i * incx] *= alpha;
}

/* Add a scaled vector to another vector */
void axpy(int n, real alpha, const real *x, int incx, real *y, int incy) {
  int i;

  for (i = 0; i < n; i++) y[i * incy] += alpha * x[i * incx];
}

/* Compute a vector's Euclidean norm */
real nrm2(int n, const real *x, int incx) {
  real val, quot, scale, iscale, sum;
  int i;

  /* Here we have to be careful to avoid overflows:
   * "scale" is the maximal absolute value encountered so far,
   * "iscale" is its reciprocal, and "sum" is the sum of (x[i] / scale)^2.
   * If we encounter a value with larger absolute value, we adjust
   * "sum", "scale", and "iscale". */
  sum = 0.0;
  scale = 0.0;
  iscale = 0.0;
  for (i = 0; i < n; i++) {
    val = ABS(x[i * incx]);

    if (val > scale) {
      iscale = 1.0 / val;
      quot = scale * iscale; /* this equals scale/val */
      sum = 1.0 + sum * quot * quot;
      scale = val;
    } else {
      val *= iscale;
      sum += val * val;
    }
  }

  return scale * SQRT(sum);
}

/* Compute the Euclidean inner product */
real dot(int n, const real *x, int incx, const real *y, int incy) {
  real sum;
  int i;

  sum = 0.0;
  for (i = 0; i < n; i++) sum += x[i * incx] * y[i * incy];

  return sum;
}

/* ------------------------------------------------------------
 * Level 2
 * ------------------------------------------------------------ */

void gemv(bool trans, int rows, int cols, real alpha, const real *A, int ldA,
          const real *x, int incx, real *y, int incy) {
  int j;

  if (trans) {
    for (j = 0; j < cols; j++)
      y[j * incy] += alpha * dot(rows, A + j * ldA, 1, x, incx);
  } else {
    for (j = 0; j < cols; j++)
      axpy(rows, alpha * x[j * incx], A + j * ldA, 1, y, incy);
  }
}

void ger(int rows, int cols, real alpha, const real *x, int incx, const real *y,
         int incy, real *A, int ldA) {
  int j;

  for (j = 0; j < cols; j++)
    axpy(rows, alpha * y[j * incy], x, incx, A + j * ldA, 1);
}

void syr(char uplo, int n, real alpha, const real *x, int incx, real *A,
         int ldA) {
  int j;

  if (uplo == 'l') {
    for (j = 0; j < n; j++) {
      axpy(n - j, alpha * x[j * incx], x + j, incx, A + j + j * ldA, 1);
    }
  } else {
    assert(uplo == 'r');
    for (j = 0; j < n; j++) {
      axpy(j + 1, alpha * x[j * incx], x, incx, A + j * ldA, 1);
    }
  }
}

void gemm(bool transA, bool transB, int rows, int cols, int k, real alpha,
          const real *A, int ldA, const real *B, int ldB, real beta, real *C,
          int ldC) {
  int j;

  if (transA) {
    if (transB) {
      for (j = 0; j < cols; j++) {
        scal(rows, beta, C + j * ldC, 1);
        gemv(transA, k, rows, alpha, A, ldA, B + j, ldB, C + j * ldC, 1);
      }
    } else {
      for (j = 0; j < cols; j++) {
        scal(rows, beta, C + j * ldC, 1);
        gemv(transA, k, rows, alpha, A, ldA, B + j * ldB, 1, C + j * ldC, 1);
      }
    }
  } else {
    if (transB) {
      for (j = 0; j < cols; j++) {
        scal(rows, beta, C + j * ldC, 1);
        gemv(transA, rows, k, alpha, A, ldA, B + j, ldB, C + j * ldC, 1);
      }
    } else {
      for (j = 0; j < cols; j++) {
        scal(rows, beta, C + j * ldC, 1);
        gemv(transA, rows, k, alpha, A, ldA, B + j * ldB, 1, C + j * ldC, 1);
      }
    }
  }
}
