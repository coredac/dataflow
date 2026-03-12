#include <cstdint>
#define N 1000

void kernel_floyd_warshall(int64_t path[N][N], int64_t i, int64_t k,
                           int64_t j) {
  // int64_t j;

  // for (k = 0; k < N; k++) {
  //   for (i = 0; i < N; i++)
  // for (j = 0; j < N; j++)
  path[i][j] = path[i][j] < path[i][k] + path[k][j] ? path[i][j]
                                                    : path[i][k] + path[k][j];
  // }
}