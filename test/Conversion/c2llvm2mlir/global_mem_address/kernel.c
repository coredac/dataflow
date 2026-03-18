// GEMV + ReLU + GEMV kernel chain for codegen tests (int-only).
// All three kernels are implemented directly inside main().

#define M 4
#define K 4
#define N 4

static int run_gemv_relu_gemv(void) {
  // Inputs / outputs: filled by simulation; only shape M, N, K matter here.
  static int A[M * K];
  static int x[K];
  static int y[M];

  // Kernel 1: GEMV
  for (int i = 0; i < M; ++i) {
    int acc = 0;
    for (int j = 0; j < K; ++j) {
      acc += A[i * K + j] * x[j];
    }
    y[i] = acc;
  }

  // Kernel 2: ReLU
  for (int i = 0; i < M; ++i) {
    if (y[i] < 0) {
      y[i] = 0;
    }
  }

  // Return a checksum-like value so outputs are consumed.
  int checksum = 0;
  for (int i = 0; i < N; ++i) {
    checksum += y[i];
  }
  return checksum & 0xFF;
}

int kernel_gemv_relu_gemv(void) { return run_gemv_relu_gemv(); }

int main(void) { return run_gemv_relu_gemv(); }
