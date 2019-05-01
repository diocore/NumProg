CC = gcc
RM = rm -f
#RM = del

CFLAGS = -Wall \
	 -O3 -g -funroll-loops -march=native -mtune=native \
         -std=c99
LIBS   = -lm
#LIBS   = -lm -lwinmm
         
CFLAGS += -DUSE_OPENMP -fopenmp

#CFLAGS += -DUSE_BLAS
#LIBS   += -lblas -lgfortran -llapack

all: p4_block_inverse

basic.o: basic.c basic.h
	$(CC) -c $(CFLAGS) $< -o $@ 
	
miniblas.o: miniblas.c miniblas.h basic.h
	$(CC) -c $(CFLAGS) $< -o $@ 
	
matrix.o: matrix.c matrix.h miniblas.h basic.h
	$(CC) -c $(CFLAGS) $< -o $@ 

inversion.o: inversion.c matrix.h miniblas.h basic.h
	$(CC) -c $(CFLAGS) $< -o $@ 
	
p4_block_inverse: p4_block_inverse.o basic.o miniblas.o matrix.o inversion.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

p4_block_inverse.o: p4_block_inverse.c matrix.h inversion.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o
	$(RM) p4_block_inverse

