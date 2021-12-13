test_mpi_polus: mpi_polus test_mpi_polus
	./test_mpi_polus

test_mpi: mpi test_mpi
	./test_mpi

mpi_polus: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c -o 2mm_mpi -std=c99

mpi: 2mm_mpi.c 2mm_mpi.h
	mpicc ./2mm_mpi.c ./2mm_mpi.h -o 2mm_mpi

test_omp_polus: omp_polus 2mm_omp
	./test_omp_polus

test_omp: omp 2mm_omp
	./test_omp

omp_polus: 2mm_omp.c 2mm_omp.h
	gcc 2mm_omp.c -o 2mm_omp -Wall -fopenmp

omp: 2mm_omp.c 2mm_omp.h
	gcc 2mm_omp.c -o 2mm_omp -Wall -fopenmp

run_simple: simple
	./2mm

simple: 2mm.c 2mm.h
	gcc 2mm.c 2mm.h -o 2mm

clean:
	rm -rf *.o 2mm
