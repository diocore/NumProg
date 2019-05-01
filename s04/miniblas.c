
#include "miniblas.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------
 * Level 1
 * ------------------------------------------------------------ */

#ifdef USE_BLAS
void dscal_(const unsigned *n, const double *alpha, double *x, const unsigned *incx);
#endif

/* Scale a vector */
void scal(unsigned int n, real alpha, real *x, unsigned int incx)
{
#ifdef USE_BLAS
  dscal_(&n, &alpha, x, &incx);
#else
  int i;

  for (i = 0; i < n; i++)
    x[i * incx] *= alpha;
#endif
}

#ifdef USE_BLAS
void daxpy_(const unsigned *n, const double *alpha, const double *x,
            const unsigned *incx, double *y, const unsigned *incy);
#endif

/* Add a scaled vector to another vector */
void axpy(unsigned int n, real alpha, const real *x, unsigned int incx,
          real *y, unsigned int incy)
{
#ifdef USE_BLAS
  daxpy_(&n, &alpha, x, &incx, y, &incy);
#else
  int i;

  if (incx != 1 || incy != 1)
  {
    for (i = 0; i < n; i++)
    {
      y[i * incy] += alpha * x[i * incx];
    }
  }
  else
  {
    for (i = 0; i < n; i++)
    {
      y[i] += alpha * x[i];
    }
  }
#endif
}

#ifdef USE_BLAS
double
dnrm2_(const unsigned *n, const double *x, const unsigned *incx);
#endif

/* Compute a vector's Euclidean norm */
real nrm2(unsigned int n, const real *x, unsigned int incx)
{
#ifdef USE_BLAS
  return dnrm2_(&n, x, &incx);
#else
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
  for (i = 0; i < n; i++)
  {
    val = ABS(x[i * incx]);

    if (val > scale)
    {
      iscale = 1.0 / val;
      quot = scale * iscale; /* this equals scale/val */
      sum = 1.0 + sum * quot * quot;
      scale = val;
    }
    else
    {
      val *= iscale;
      sum += val * val;
    }
  }

  return scale * SQRT(sum);
#endif
}

#ifdef USE_BLAS
double
ddot_(const unsigned *n, const double *x, const unsigned *incx, const double *y,
      const unsigned *incy);
#endif

/* Compute the Euclidean inner product */
real dot(unsigned int n, const real *x, unsigned int incx, const real *y, unsigned int incy)
{
#ifdef USE_BLAS
  return ddot_(&n, x, &incx, y, &incy);
#else
  real sum;
  int i;

  sum = 0.0;
  for (i = 0; i < n; i++)
    sum += x[i * incx] * y[i * incy];

  return sum;

#endif
}

/* ------------------------------------------------------------
 * Level 2
 * ------------------------------------------------------------ */

#ifdef USE_BLAS
void dgemv_(const char *trans, const unsigned *m, const unsigned *n,
            const double *alpha, const double *a, const unsigned *lda, const double *x,
            const unsigned *incx, const double *beta, double *y, const unsigned *incy);
#endif

void gemv(bool trans, unsigned int rows, unsigned int cols, real alpha,
          const real *A, unsigned int ldA,
          const real *x, unsigned int incx,
          real *y, unsigned int incy)
{
#ifdef USE_BLAS
  double beta = 1.0;
  dgemv_(trans ? "TRANS" : "Not Trans", &rows, &cols, &alpha, A, &ldA, x,
         &incx, &beta, y, &incy);
#else
  int j;

  if (trans)
  {
    for (j = 0; j < cols; j++)
      y[j * incy] += alpha * dot(rows, A + j * ldA, 1, x, incx);
  }
  else
  {
    for (j = 0; j < cols; j++)
      axpy(rows, alpha * x[j * incx], A + j * ldA, 1, y, incy);
  }
#endif
}

#ifdef USE_BLAS
void dger_(const unsigned *m, const unsigned *n, const double *alpha,
           const double *x, const unsigned *incx, const double *y,
           const unsigned *incy, double *a, const unsigned *lda);
#endif

void ger(unsigned int rows, unsigned int cols, real alpha,
         const real *x, unsigned int incx,
         const real *y, unsigned int incy,
         real *A, unsigned int ldA)
{
#ifdef USE_BLAS
  dger_(&rows, &cols, &alpha, x, &incx, y, &incy, A, &ldA);
#else
  int j;

  for (j = 0; j < cols; j++)
    axpy(rows, alpha * y[j * incy], x, incx, A + j * ldA, 1);
#endif
}

#ifdef USE_BLAS
void dsyr_(const char *uplo, const unsigned *n, const double *alpha,
           const double *x, const unsigned *incx,
           double *a, const unsigned *lda);
#endif

void syr(char uplo, unsigned int n, real alpha,
         const real *x, unsigned int incx,
         real *A, unsigned int ldA)
{
#ifdef USE_BLAS
  dsyr_(uplo, &n, &alpha, x, &incx, A, &ldA);
#else
  int j;

  if (uplo == 'l')
  {
    for (j = 0; j < n; j++)
    {
      axpy(n - j, alpha * x[j * incx], x + j, incx, A + j + j * ldA, 1);
    }
  }
  else
  {
    assert(uplo == 'r');
    for (j = 0; j < n; j++)
    {
      axpy(j + 1, alpha * x[j * incx], x, incx, A + j * ldA, 1);
    }
  }
#endif
}

#ifdef USE_BLAS
void dgemm_(const char *transa, const char *transb, const unsigned *m,
            const unsigned *n, const unsigned *k, const double *alpha, const double *a,
            const unsigned *lda, const double *b, const unsigned *ldb,
            const double *beta, double *c, const unsigned *ldc);
#endif

void gemm(bool transA, bool transB, unsigned int rows, unsigned int cols, unsigned int k, real alpha,
          const real *A, unsigned int ldA,
          const real *B, unsigned int ldB,
          real beta,
          real *C, unsigned int ldC)
{
#ifdef USE_BLAS
  dgemm_(transA ? "Trans" : "Not Trans", transB ? "Trans" : "Not Trans", &rows, &cols, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC);
#else
  int j;

  if (transA)
  {
    if (transB)
    {
      for (j = 0; j < cols; j++)
      {
        scal(rows, beta, C + j * ldC, 1);
        gemv(transA, k, rows, alpha, A, ldA, B + j, ldB, C + j * ldC, 1);
      }
    }
    else
    {
      for (j = 0; j < cols; j++)
      {
        scal(rows, beta, C + j * ldC, 1);
        gemv(transA, k, rows, alpha, A, ldA, B + j * ldB, 1, C + j * ldC, 1);
      }
    }
  }
  else
  {
    if (transB)
    {
      for (j = 0; j < cols; j++)
      {
        scal(rows, beta, C + j * ldC, 1);
        gemv(transA, rows, k, alpha, A, ldA, B + j, ldB, C + j * ldC, 1);
      }
    }
    else
    {
      for (j = 0; j < cols; j++)
      {
        scal(rows, beta, C + j * ldC, 1);
        gemv(transA, rows, k, alpha, A, ldA, B + j * ldB, 1, C + j * ldC, 1);
      }
    }
  }
#endif
}
