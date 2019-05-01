CC = gcc
RM = rm -f
#RM = del

CFLAGS = -Wall \
	 -O3 -funroll-loops -march=native -mtune=native \
         -std=c99
LIBS   = -lm
#LIBS   = -lm -lwinmm
         
CFLAGS += -DUSE_OPENMP -fopenmp

# CFLAGS += -DUSE_BLAS
# LIBS   += -lblas -lgfortran -llapack

all: p1_blas_ldlt

basic.o: basic.c basic.h
	$(CC) -c $(CFLAGS) $< -o $@ 
	
miniblas.o: miniblas.c miniblas.h basic.h
	$(CC) -c $(CFLAGS) $< -o $@ 
	
matrix.o: matrix.c matrix.h miniblas.h basic.h
	$(CC) -c $(CFLAGS) $< -o $@ 
	
p1_blas_ldlt: p1_blas_ldlt.o basic.o miniblas.o matrix.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

p1_blas_ldlt.o: p1_blas_ldlt.c matrix.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o
	$(RM) p1_blas_ldlt

