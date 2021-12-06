run_omp: omp
	./2mm_omp
omp: 2mm_omp.c 2mm.h
	gcc 2mm_omp.c 2mm.h -o 2mm_omp -fopenmp

run_simple: simple
	./2mm
simple: 2mm.c 2mm.h
	gcc 2mm.c 2mm.h -o 2mm

clean:
	rm -rf *.o 2mm
