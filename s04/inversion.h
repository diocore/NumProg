#ifndef INVERSION_H
#define INVERSION_H

#include <assert.h>

/* Additional libraries */
#include "basic.h"    /* basic types and time measurement */
#include "miniblas.h" /* our simple BLAS version */
#include "matrix.h"   /* matrix functions */

void invert(pmatrix a);
int invert_block(pmatrix a, int blocks, int resolution);

#endif