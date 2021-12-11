/* Include benchmark-specific header. */
#include "2mm_omp.h"

double bench_t_start, bench_t_end;
char* file_path;

static double rtclock() {
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, NULL);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
  bench_t_start = rtclock();
}

void bench_timer_stop() {
  bench_t_end = rtclock();
}

void bench_timer_print() {
  FILE* fout;
  fout = fopen(file_path, "a+");
  fprintf(fout, "%0.6lf;", bench_t_end - bench_t_start);
  printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

static void init_array(int ni,
                       int nj,
                       int nk,
                       int nl,
                       float* alpha,
                       float* beta,
                       float A[ni][nk],
                       float B[nk][nj],
                       float C[nj][nl],
                       float D[ni][nl]) {
  int i, j;

  *alpha = 1.0;
  *beta = 1.0;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (float)((i * j + 1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (float)(i * (j + 1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (float)((i * (j + 3) + 1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (float)(i * (j + 2) % nk) / nk;
}

static void print_array(int ni, int nl, float D[ni][nl]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "D");
  for (i = 0; i < ni; i++) {
    fprintf(stderr, "\n");
    for (j = 0; j < nl; j++) {
      fprintf(stderr, "%0.2f ", D[i][j]);
    }
  }
  fprintf(stderr, "\nend   dump: %s\n", "D");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_2mm(int ni,
                       int nj,
                       int nk,
                       int nl,
                       float alpha,
                       float beta,
                       float tmp[ni][nj],
                       float A[ni][nk],
                       float B[nk][nj],
                       float C[nj][nl],
                       float D[ni][nl]) {
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
#pragma omp master
    {
      FILE* fout;
      fout = fopen(file_path, "a+");
      printf("Nthread = %d\n", nthreads);
      fprintf(fout, "%d\n", nthreads);
    }
    int i, j, k;
    for (i = 0; i < ni; i++)
      for (j = 0; j < nj; j++)
        tmp[i][j] = 0.0f;
#pragma omp barrier
#pragma omp for
    for (i = 0; i < ni; i++)
      for (k = 0; k < nk; ++k)
        for (j = 0; j < nj; j++)
          tmp[i][j] += alpha * A[i][k] * B[k][j];
#pragma omp for
    for (i = 0; i < ni; i++)
      for (j = 0; j < nl; j++)
        D[i][j] *= beta;
#pragma omp barrier
#pragma omp for
    for (i = 0; i < ni; i++)
      for (k = 0; k < nj; ++k)
        for (j = 0; j < nl; j++)
          D[i][j] += tmp[i][k] * C[k][j];
  }
}

int main(int argc, char** argv) {

  if (argc == 1){
    file_path = "~/skipod/skipod_polus/2mm1/result_omp.csv";
  } else {
    file_path = argv[1];
  }

  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  float alpha;
  float beta;
  float(*tmp)[ni][nj];
  tmp = (float(*)[ni][nj])malloc((ni) * (nj) * sizeof(float));
  float(*A)[ni][nk];
  A = (float(*)[ni][nk])malloc((ni) * (nk) * sizeof(float));
  float(*B)[nk][nj];
  B = (float(*)[nk][nj])malloc((nk) * (nj) * sizeof(float));
  float(*C)[nj][nl];
  C = (float(*)[nj][nl])malloc((nj) * (nl) * sizeof(float));
  float(*D)[ni][nl];
  D = (float(*)[ni][nl])malloc((ni) * (nl) * sizeof(float));

  init_array(ni, nj, nk, nl, &alpha, &beta, *A, *B, *C, *D);

  /*printf("A:\n");
  print_array(ni, nk, *A);

  printf("B:\n");
  print_array(nk, nj, *B);

  printf("C:\n");
  print_array(nj, nl, *C);

  printf("D:\n");
  print_array(ni, nl, *D);*/

  bench_timer_start();

  kernel_2mm(ni, nj, nk, nl, alpha, beta, *tmp, *A, *B, *C, *D);

  bench_timer_stop();
  bench_timer_print();

  if (argc > 42 && !strcmp(argv[0], ""))
    print_array(ni, nl, *D);

  free((void*)tmp);
  free((void*)A);
  free((void*)B);
  free((void*)C);
  free((void*)D);

  return 0;
}
