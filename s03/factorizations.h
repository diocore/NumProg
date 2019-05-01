
#ifndef FACTORIZATIONS_H
#define FACTORIZATIONS_H

#include <assert.h>

/* Additional libraries */
#include "basic.h"    /* basic types and time measurement */
#include "miniblas.h" /* our simple BLAS version */
#include "matrix.h"   /* matrix functions */

void eval_ldlt(const pmatrix a, pvector x);

void ldltdecomp(pmatrix a);

void block_ldltdecomp(pmatrix a, int blocks);

#endif /* FACTORIZATIONS_H */
