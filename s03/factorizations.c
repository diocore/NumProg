/* ------------------------------------------------------------
 * RL decomposition using BLAS
 * ------------------------------------------------------------ */

#include "factorizations.h"
#include <stdio.h>

static void eval_l(const pmatrix a, pvector x) {
  int n = a->rows;
  int lda = a->ld;
  double *aa = a->a;
  double *xx = x->x;
  int k;

  assert(a->cols == n);
  assert(x->rows == n);

  for (k = n; k-- > 0;) {
    axpy(n - k - 1, xx[k], aa + k + 1 + k * lda, 1, xx + k + 1, 1);
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

  for (k = 0; k < n; k++) {
    xx[k] *= aa[k + k * lda];
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

  for (k = 0; k < n; k++) {
    xx[k] += dot(n - k - 1, aa + (k + 1) + k * lda, 1, xx + k + 1, 1);
  }
}

void eval_ldlt(const pmatrix a, pvector x) {
  eval_lt(a, x);
  eval_d(a, x);
  eval_l(a, x);
}

void ldltdecomp(pmatrix a) {
  int lda = a->ld;
  int n = a->rows;
  double *aa = a->a;
  int k;

  assert(a->cols == n);

  for (k = 0; k < n; k++) {
    scal(n - k - 1, 1.0 / aa[k + k * lda], aa + (k + 1) + k * lda, 1);
    syr('l', n - k - 1, -aa[k + k * lda], aa + (k + 1) + k * lda, 1,
        aa + (k + 1) + (k + 1) * lda, lda);
  }
}

void block_lsolve_trans(pmatrix l, pmatrix x) {
  int n = l->rows;
  int m = x->rows;
  int ldl = l->ld;
  int ldx = x->ld;
  double *ll = l->a;
  double *xx = x->a;

  int k;

  assert(l->cols == n);
  assert(x->cols == n);

  for (k = 0; k < n; k++) {
    ger(m, n - k - 1, -1.0, xx + k * ldx, 1, ll + (k + 1) + k * ldl, 1,
        xx + (k + 1) * ldx, ldx);
  }
}

void block_dsolve_trans(pmatrix d, pmatrix x) {
  int n = d->rows;
  int m = x->rows;
  int ldd = d->ld;
  int ldx = x->ld;
  double *dd = d->a;
  double *xx = x->a;

  assert(d->cols == n);
  assert(x->cols == n);

  int k;

  for (k = 0; k < n; k++) {
    scal(m, 1.0 / dd[k + k * ldd], xx + k * ldx, 1);
  }
}

void addmul(double alpha, bool atrans, pmatrix a, bool btrans, pmatrix b,
            pmatrix c) {
  if (atrans) {
    if (btrans) {
      assert(a->cols == c->rows);
      assert(a->rows == b->cols);
      assert(b->rows == c->cols);
      gemm(true, true, c->rows, c->cols, a->rows, alpha, a->a, a->ld, b->a,
           b->ld, 1.0, c->a, c->ld);
    } else {
      assert(a->cols == c->rows);
      assert(a->rows == b->rows);
      assert(b->cols == c->cols);
      gemm(true, false, c->rows, c->cols, a->rows, alpha, a->a, a->ld, b->a,
           b->ld, 1.0, c->a, c->ld);
    }
  } else {
    if (btrans) {
      assert(a->rows == c->rows);
      assert(a->cols == b->cols);
      assert(b->rows == c->cols);
      gemm(false, true, c->rows, c->cols, a->cols, alpha, a->a, a->ld, b->a,
           b->ld, 1.0, c->a, c->ld);
    } else {
      assert(a->rows == c->rows);
      assert(a->cols == b->rows);
      assert(b->cols == c->cols);
      gemm(false, false, c->rows, c->cols, a->cols, alpha, a->a, a->ld, b->a,
           b->ld, 1.0, c->a, c->ld);
    }
  }
}

void block_ldltdecomp(pmatrix a, int blocks) {
  // TODO
  // the code is working for our test matrix of size 4x4 and a block size of 2
  // with the same error as the reference function in the test. For some reason
  // we get a huge error for a hilbert matrix of the same size with the same
  // block size. We don't know why and it is hard to test since we can;t just
  // build huge matrixes by hand or check the values. We know that our code does
  // not work for some (or many) other sizes of matrices. But since we don;t
  // even know why a 4x4 hilbert matrix does not work and a normal one (the 4x4
  // of the previous exercise) we were unable to find the error

  // there is also no cleanup of the matrixes created with new_sub_matrix but
  // since it doesn't really work why bother freeing the memory

  pmatrix sub;
  pmatrix Xt;
  pmatrix C = new_zero_matrix(blocks, blocks);
  pmatrix diagm;

  for (int i = 0; i < a->cols / blocks; i++) {
    // printf("======================\n");
    // print_matrix(a);
    // printf("======================\n");

    sub = new_sub_matrix(a, blocks, i * blocks, blocks, i * blocks);
    ldltdecomp(sub);

    for (int k = i + 1; k < a->cols / blocks; k++) {
      Xt = new_sub_matrix(a, blocks, k * blocks, blocks, i * blocks);
      block_dsolve_trans(sub, Xt);
      block_lsolve_trans(sub, Xt);
    }

    for (int k = i + 1; k < a->cols / blocks; k++) {
      for (int j = i + 1; j < a->cols / blocks; j++) {
        pmatrix l =
            new_sub_matrix(a, blocks, k * blocks, blocks, (j - 1) * blocks);
        diagm = new_zero_matrix(blocks, blocks);

        for (int i = 0; i < blocks; i++) {
          diagm->a[i + i * diagm->ld] =
              sub->a[i + i * sub->ld];  // only way how we got this to work
        }

        Xt = new_sub_matrix(a, blocks, k * blocks, blocks, k * blocks);
        addmul(1, false, diagm, true, l, C);
        addmul(-1, false, l, false, C, Xt);
      }
    }
  }
}
