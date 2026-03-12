#include <cstdint>
#define N 2000
#define TSTEPS 500

void jacobi(double A[N], double B[N], int64_t t) {
  int64_t i;

  // for (t = 0; t < TSTEPS; t++) {
  // for (i = 1; i < N - 1; i++)
  //   B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
  for (i = 1; i < N - 1; i++)
    A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1]);
  // }
}