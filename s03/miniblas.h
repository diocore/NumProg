#ifndef MINIBLAS_H
#define MINIBLAS_H

#include "basic.h"

/** @defgroup miniblas miniblas
 *  @brief Minimal BLAS implementation for teaching purposes.
 *
 *  Vectors are represented by arrays of coefficients and increments,
 *  i.e., we have @f$x_i=\texttt{x[i*incx]}@f$.
 *
 *  Matrices are represented in column-major order by arrays of
 *  coefficients and leading dimensions,
 *  i.e., we have @f$a_{ij} = \texttt{A[i+j*ldA]}@f$.
 *
 *  @author Steffen B&ouml;rm, <sb@informatik.uni-kiel.de>
 *
 *  @{ */

/* ------------------------------------------------------------
 * Level 1: Vector-vector operations
 * ------------------------------------------------------------ */

/** @brief Scale a vector, @f$x \gets \alpha x@f$.
 *
 *  @param n Dimension of @f$x@f$.
 *  @param alpha Scaling factor @f$\alpha@f$.
 *  @param x Coefficients of @f$x@f$.
 *  @param incx Increment of @f$x@f$. */
HEADER_PREFIX void
scal(unsigned int n, real alpha, real *x, unsigned int incx);

/** @brief Add a scaled vector, @f$y \gets \alpha x + y@f$.
 *
 *  @param n Dimension of @f$x@f$ and @f$y@f$.
 *  @param alpha Scaling factor @f$\alpha@f$.
 *  @param x Coefficients of @f$x@f$.
 *  @param incx Increment of @f$x@f$.
 *  @param y Coefficients of @f$y@f$.
 *  @param incy Increment of @f$y@f$. */
HEADER_PREFIX void
axpy(unsigned int n, real alpha, const real *x, unsigned int incx,
     real *y, unsigned int incy);

/** @brief Euclidean norm, @f$\|x\|_2 = \sqrt{\langle x, x\rangle_2}@f$.
 *
 *  @param n Dimension of @f$x@f$.
 *  @param x Coefficients of @f$x@f$.
 *  @param incx Increment of @f$x@f$.
 *  @returns Euclidean norm @f$\|x\|_2@f$. */
HEADER_PREFIX real
nrm2(unsigned int n, const real *x, unsigned int incx);

/** @brief Euclidean inner product,
 *    @f$\langle x,y \rangle_2 = \sum_{i=1}^n \bar x_i y_i@f$.
 *
 *  @param n Dimension of @f$x@f$ and @f$y@f$.
 *  @param x Coefficients of @f$x@f$.
 *  @param incx Increment of @f$x@f$.
 *  @param y Coefficients of @f$y@f$.
 *  @param incy Increment of @f$y@f$.
 *  @returns Euclidean inner product @f$\langle x, y \rangle_2@f$. */
HEADER_PREFIX real
dot(unsigned int n, const real *x, unsigned int incx, const real *y, unsigned int incy);

/* ------------------------------------------------------------
 * Level 2: Matrix-vector operations
 * ------------------------------------------------------------ */

/** @brief Matrix-vector product, @f$y \gets y + \alpha A x@f$.
 *
 *  @param trans Do we want to use the adjoint @f$A^*@f$ instead of @f$A@f$?
 *  @param rows Rows of @f$A@f$.
 *  @param cols Columns of @f$A@f$.
 *  @param alpha Scaling factor @f$\alpha@f$.
 *  @param A Coefficients of @f$A@f$.
 *  @param ldA Leading dimension of @f$A@f$.
 *  @param x Coefficients of @f$x@f$.
 *  @param incx Increment of @f$x@f$.
 *  @param y Coefficients of @f$y@f$.
 *  @param incy Increment of @f$y@f$. */
HEADER_PREFIX void
gemv(bool trans, unsigned int rows, unsigned int cols,
     real alpha, const real *A, unsigned int ldA,
     const real *x, unsigned int incx,
     real *y, unsigned int incy);

/** @brief Rank-one update, @f$A \gets A + \alpha x y^*@f$.
 *
 *  @param rows Rows of @f$A@f$, dimension of @f$x@f$.
 *  @param cols Columns of @f$A@f$, dimension of @f$y@f$.
 *  @param alpha Scaling factor @f$\alpha@f$.
 *  @param x Coefficients of @f$x@f$.
 *  @param incx Increment of @f$x@f$.
 *  @param y Coefficients of @f$y@f$.
 *  @param incy Increment of @f$y@f$.
 *  @param A Coefficients of @f$A@f$.
 *  @param ldA Leading dimension of @f$A@f$. */
HEADER_PREFIX void
ger(unsigned int rows, unsigned int cols, real alpha,
    const real *x, unsigned int incx,
    const real *y, unsigned int incy,
    real *A, unsigned int ldA);

/** @brief Symmetric rank-one update, @f$A \gets A + \alpha x x^*@f$.
 *
 *  @param uplo Character determining if only the 'l'ower triangular
 *         part or only the 'r'ight triangular part should be 
 *         updated.
 *  @param n Rows and columns of @f$A@f$, dimension of @f$x@f$.
 *  @param alpha Scaling factor @f$\alpha@f$.
 *  @param x Coefficients of @f$x@f$.
 *  @param incx Increment of @f$x@f$.
 *  @param A Coefficients of @f$A@f$.
 *  @param ldA Leading dimension of @f$A@f$. */
HEADER_PREFIX void
syr(char uplo, unsigned int n, real alpha,
    const real *x, unsigned int incx,
    real *A, unsigned int ldA);

/* ------------------------------------------------------------
 * Level 3: Matrix-matrix operations
 * ------------------------------------------------------------ */

/** @brief Matrix-matrix product, @f$C \gets \alpha A B + \beta C@f$.
 *
 *  @param transA Do we want to use the adjoint @f$A^*@f$ instead of @f$A@f$?
 *  @param transB Do we want to use the adjoint @f$B^*@f$ instead of @f$B@f$?
 *  @param rows Rows of @f$A@f$ and @f$C@f$.
 *  @param cols Columns of @f$B@f$ and @f$C@f$.
 *  @param k Columns of @f$A@f$ and rows of @f$B@f$.
 *  @param alpha Scaling factor @f$\alpha@f$.
 *  @param A Coefficients of @f$A@f$.
 *  @param ldA Leading dimension of @f$A@f$.
 *  @param B Coefficients of @f$B@f$.
 *  @param ldB Leading dimension of @f$B@f$.
 *  @param beta Scaling factor @f$\beta@f$.
 *  @param C Coefficients of @f$C@f$.
 *  @param ldC Leading dimension of @f$C@f$. */
HEADER_PREFIX void
gemm(bool transA, bool transB, unsigned int rows, unsigned int cols, unsigned int k, real alpha,
     const real *A, unsigned int ldA,
     const real *B, unsigned int ldB,
     real beta,
     real *C, unsigned int ldC);

/** @} */

#endif
