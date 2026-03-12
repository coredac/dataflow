#include <cstdint>
#define NNZ 1666
#define N 494
#define L 10

void spmv(int64_t nzval[N * L], int64_t cols[N * L], int64_t vec[N],
          int64_t out[N], int64_t i, int64_t j) {
  // int64_t j;
  int64_t Si;

  // for (i = 0; i < N; i++) {
  int64_t sum = out[i];
  // for (j = 0; j < L; j++) {
  Si = nzval[j + i * L] * vec[cols[j + i * L]];
  sum += Si;
  // }
  out[i] = sum;
  // }
}