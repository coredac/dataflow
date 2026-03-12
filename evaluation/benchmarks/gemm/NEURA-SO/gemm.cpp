#include <cstdint>
#define row_size 64
#define col_size 64
#define N row_size *col_size

void gemm(int64_t m1[N], int64_t m2[N], int64_t prod[N], int64_t j,
          int64_t idx) {
  // for (int64_t idx = 0; idx < N; idx++) {
  int64_t i = idx / col_size;
  // int64_t j = idx % col_size;
  int64_t i_col = i * col_size;

  for (int64_t k = 0; k < row_size; k++) {
    int64_t k_col = k * col_size;
    prod[idx] += m1[i_col + k] * m2[k_col + j];
  }
  // }
}