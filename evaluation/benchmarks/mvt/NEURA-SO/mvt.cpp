#include <cstdint>
#define N 2000

void mvt(int n, int64_t x1[N], int64_t x2[N], int64_t y_1[N], int64_t y_2[N],
         int64_t A[N][N]) {
  // int i;
  int64_t j = 0;
  // for (j = 0; j < N; j++) {
  for (int64_t i = 0; i < N; i++) {
    x1[i] = x1[i] + A[i][j] * y_1[j];
    x2[i] = x2[i] + A[j][i] * y_2[j];
  }
  // }
}