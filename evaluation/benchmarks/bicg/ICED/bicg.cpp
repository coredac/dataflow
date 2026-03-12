#include <cstdint>
#define N 2100
#define M 1900
#define DATA_TYPE int64_t

void bicg(DATA_TYPE A[N][M], DATA_TYPE s[M], DATA_TYPE q[N], DATA_TYPE p[M],
          DATA_TYPE r[N]) {
  int64_t i = 3;
  // for (int i = 0; i < N; i++) {
  q[i] = 0.0;

  for (int j = 0; j < M; j++) {
    s[j] += r[i] * A[i][j];
    q[i] += A[i][j] * p[j];
  }
  // }
}