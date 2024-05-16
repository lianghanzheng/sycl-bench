#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

typedef double DataT;

DataT *in;
DataT *out;
size_t problem_size = 524288;

double timer() {
  struct timeval t_val;
  gettimeofday(&t_val, NULL);
  double time_in_sec = t_val.tv_sec + t_val.tv_usec*1e-6;
  return time_in_sec;
}

void setup() {
  in = (DataT *)malloc(problem_size*sizeof(DataT));
  out = (DataT *)malloc(problem_size*sizeof(DataT));

  for (size_t i = 0; i < problem_size; ++i) in[i] = 1.0;
}

void arithKernel() {
# pragma omp parallel for
  for (size_t k = 0; k < problem_size; ++k) {
    DataT a1 = in[k];
    const DataT a2 = a1;

    for (size_t i = 0; i < 512; ++i) {
      a1 = a1 * a1 + a1;
      a1 = a1 * a2 - a2;
    }

    out[k] = a1;
  }
}

int verify() {
  for (size_t i = 0; i < problem_size; ++i) {
    if (out[i] != 1.0) return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Using default problem size\n");
  } else {
    problem_size = strtol(argv[1], NULL, 10);
  }

  setup();

  double time[5];
  double start, end;

  for (size_t i = 0; i < 5; ++i) {
    start = timer();
    arithKernel();
    end = timer();
    if (!verify()) {
      fprintf(stderr, "Wrong result\n");
      exit(1);
    }
    time[i] = end - start;
  }

  printf("Time Samples: %.6f, %.6f, %.6f, %.6f, %.6f\n",
    time[0], time[1], time[2], time[3], time[4]);

  double throughput = (double)problem_size * 512 * 2 * 2 /
                      1024.0 / 1024.0 / 1024.0;
  printf("  -> Throughput = %f GOP\n", throughput);
  
  double min_time = 999.9f;
  for (size_t i = 0; i < 5; ++i) {
    min_time = time[i] < min_time ? time[i] : min_time;
  }
  printf("  -> GOPs = %f\n", throughput / min_time);

  free(in); 
  free(out);  
}