#include <cstdint>
#define NNZ 1666
#define N 494
#define L 10

void spmv(int64_t nzval[N * L], int64_t cols[N * L], int64_t vec[N],
          int64_t out[N]) {
  // int64_t j;
  // int64_t Si;
  // int64_t i = 3;
  // for (i = 0; i < N; i++) {
  int64_t sum = 0;
  for (int64_t j = 0; j < L; j++) {
    sum += nzval[j + 3 * L] * vec[cols[j + 3 * L]];
    // sum += Si;
  }
  out[3] = sum;
  // }
}