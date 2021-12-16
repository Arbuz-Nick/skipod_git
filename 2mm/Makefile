
test_mpi_bg: mpi_bg test_mpi_bg
	./test_mpi_bg

test_mpi_polus: mpi_polus test_mpi_polus
	./test_mpi_polus

test_mpi: test_mpi
	./test_mpi

mpi_bg: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c 2mm_mpi.h -o 2mm_mpi -std=c99

mpi_xs: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c 2mm_mpi.h -o 2mm_mpi_xs -std=c99 -DMINI_DATASET

mpi_s: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c 2mm_mpi.h -o 2mm_mpi_s -std=c99 -DSMALL_DATASET

mpi_m: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c 2mm_mpi.h -o 2mm_mpi_m -std=c99 -DMEDIUM_DATASET

mpi_l: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c 2mm_mpi.h -o 2mm_mpi_l -std=c99 -DLARGE_DATASET

mpi_xl: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c 2mm_mpi.h -o 2mm_mpi_xl -std=c99 -DEXTRALARGE_DATASET

mpi: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c ./2mm_mpi.h -o 2mm_mpi

test_omp_polus: test_omp_polus
	./test_omp_polus

test_omp: omp 2mm_omp
	./test_omp

omp_polus_mini: 2mm_omp.c 2mm_omp.h
	gcc 2mm_omp.c -o 2mm_omp_xs -Wall -fopenmp -DMINI_DATASET

omp_polus_small: 2mm_omp.c 2mm_omp.h
	gcc 2mm_omp.c -o 2mm_omp_s -Wall -fopenmp -DSMALL_DATASET

omp_polus_medium: 2mm_omp.c 2mm_omp.h
	gcc 2mm_omp.c -o 2mm_omp_m -Wall -fopenmp -DMEDIUM_DATASET

omp_polus_large: 2mm_omp.c 2mm_omp.h
	gcc 2mm_omp.c -o 2mm_omp_l -Wall -fopenmp -DLARGE_DATASET

omp_polus_xlarge: 2mm_omp.c 2mm_omp.h
	gcc 2mm_omp.c -o 2mm_omp_xl -Wall -fopenmp -DEXTRALARGE_DATASET

omp: 2mm_omp.c 2mm_omp.h
	gcc 2mm_omp.c -o 2mm_omp -Wall -fopenmp

run_simple: simple
	./2mm

simple: 2mm.c 2mm.h
	gcc 2mm.c 2mm.h -o 2mm

clean:
	rm -rf *.o 2mm
