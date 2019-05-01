/* ------------------------------------------------------------
 * Matrix inversion using BLAS
 * ------------------------------------------------------------ */

#include "inversion.h"

static void addmul(double alpha, bool atrans, pmatrix a, bool btrans, pmatrix b,
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

void invert(pmatrix a) {
  pmatrix b;
  int i, j, k, n, lda;
  double *aa, *ab;

  lda = a->ld;
  n = a->rows;
  assert(n == a->cols);
  b = new_zero_matrix(n, n);
  aa = a->a;
  ab = b->a;

  for (k = 0; k < n; k++) {
    aa[k + k * lda] = 1.0 / aa[k + k * lda];
    for (i = k + 1; i < n; i++) {
      ab[i + k * n] = aa[i + k * lda] * aa[k + k * lda];
    }
    for (i = k + 1; i < n; i++) {
      ab[k + i * n] = aa[k + k * lda] * aa[k + i * lda];
    }
    for (i = k + 1; i < n; i++) {
      for (j = k + 1; j < n; j++) {
        aa[i + j * lda] -= ab[i + k * n] * aa[k + j * lda];
      }
    }
  }
  for (k = n; k-- > 0;) {
    for (i = k + 1; i < n; i++) {
      aa[i + k * lda] = 0.0;
      for (j = k + 1; j < n; j++) {
        aa[i + k * lda] -= aa[i + j * lda] * ab[j + k * n];
      }
      aa[k + i * lda] = 0.0;
      for (j = k + 1; j < n; j++) {
        aa[k + i * lda] -= ab[k + j * n] * aa[j + i * lda];
      }
    }
    for (i = k + 1; i < n; i++) {
      aa[k + k * lda] -= ab[k + i * n] * aa[i + k * lda];
    }
  }

  del_matrix(b);
}

int invert_block(pmatrix a, int blocks, int resolution) {
  pmatrix b, bkk, akk, aij, aik, akj, bik, bki, bkj, bjk;
  int i, j, k, n, minblocksize, blocksize;
  blocksize = a->cols / blocks;
  n = a->rows;
  assert(n == a->cols);
  b = new_zero_matrix(n, n);
  blocksize = n / blocks;
  minblocksize = 2;

  akk = new_sub_matrix(a, blocksize, 0, blocksize, 0);
  if (blocksize > resolution) {
    invert_block(akk, blocks, resolution);
  } else {
    invert(akk);
  }
  
  #pragma omp parallel 
  {
    #pragma omp single 
    {
      #pragma omp task
      {
        for (j = k + 1; j < blocks; j++) {
          akj = new_sub_matrix(a, blocksize, k * blocksize, blocksize, j * blocksize);
          bkj = new_sub_matrix(b, blocksize, k * blocksize, blocksize, j * blocksize);
          addmul(1, false, akk, false, akj, bkj);

          del_sub_matrix(akj);
          del_sub_matrix(bkj);
        }
      }

      #pragma omp task
      {
        for (i = k + 1; i < blocks; i++) {
          aik = new_sub_matrix(a, blocksize, i * blocksize, blocksize, k * blocksize);
          bik = new_sub_matrix(b, blocksize, i * blocksize, blocksize, k * blocksize);
          addmul(1, false, aik, false, akk, bik);

          del_sub_matrix(aik);
          del_sub_matrix(bik);
        }
      }
    }
  }

  for (k = 1; k < blocks; k++) {
    akk = new_sub_matrix(a, blocksize, k * blocksize, blocksize, k * blocksize);
    // (iv) Update von A22 zu S := A22−A21A−1 11 A12 = A22−B21A12.
    i = 1;
    j = 1;
    bik = new_sub_matrix(b, blocksize, i * blocksize, blocksize,
                         (k - 1) * blocksize);
    akj = new_sub_matrix(a, blocksize, (k - 1) * blocksize, blocksize,
                         j * blocksize);
    addmul(-1, false, bik, false, akj, akk);

    del_sub_matrix(akj);
    del_sub_matrix(bik);

    if (blocksize > resolution) {
      invert_block(akk, blocks, resolution);
    } else {
      invert(akk);
    }

    // (vi) Update von A21 zu−S−1B21.
    for (i = k; i < blocks; i++) {
      aik = new_sub_matrix(a, blocksize, i * blocksize, blocksize,
                           (k - 1) * blocksize);
      bik = new_sub_matrix(b, blocksize, i * blocksize, blocksize,
                           (k - 1) * blocksize);
      clear_matrix(aik);
      addmul(-1, false, akk, false, bik, aik);

      del_sub_matrix(aik);
      del_sub_matrix(bik);
    }

    // (vii) Update von A12 zu −B_12S^−1
    for (j = k; j < blocks; j++) {
      akj = new_sub_matrix(a, blocksize, (k - 1) * blocksize, blocksize,
                           j * blocksize);
      bkj = new_sub_matrix(b, blocksize, (k - 1) * blocksize, blocksize,
                           j * blocksize);
      clear_matrix(akj);
      addmul(-1, false, bkj, false, akk, akj);

      del_sub_matrix(aik);
      del_sub_matrix(bik);
    }

    j = 1;
    i = 1;
    bkk = new_sub_matrix(b, blocksize, (k - 1) * blocksize, blocksize,
                         (k - 1) * blocksize);
    bkj = new_sub_matrix(b, blocksize, (k - 1) * blocksize, blocksize,
                         j * blocksize);
    bik = new_sub_matrix(b, blocksize, i * blocksize, blocksize,
                         (k - 1) * blocksize);

    addmul(1, false, akk, false, bik, bkk);

    akk = new_sub_matrix(a, blocksize, (k - 1) * blocksize, blocksize,
                         (k - 1) * blocksize);
    addmul(1, false, bkj, false, bkk, akk);

    //(ii) B12 := (A_11)^-1 A12
    for (j = k + 1; j < blocks; j++) {
      akj =
          new_sub_matrix(a, blocksize, k * blocksize, blocksize, j * blocksize);
      bkj =
          new_sub_matrix(b, blocksize, k * blocksize, blocksize, j * blocksize);
      addmul(1, false, akk, false, akj, bkj);

      del_sub_matrix(akj);
      del_sub_matrix(bkj);
    }
    //(iii) B21 := A21 (A_11)^-1
    for (i = k + 1; i < blocks; i++) {
      aik =
          new_sub_matrix(a, blocksize, i * blocksize, blocksize, k * blocksize);
      bik =
          new_sub_matrix(b, blocksize, i * blocksize, blocksize, k * blocksize);
      addmul(1, false, aik, false, akk, bik);

      del_sub_matrix(aik);
      del_sub_matrix(bik);
    }
  }

  // printf("==========================================\n");
  // print_matrix(a);
  del_sub_matrix(akk);
  return minblocksize;
}
