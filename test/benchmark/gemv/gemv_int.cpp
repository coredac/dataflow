// A simple int GEMV: y = A*x + y.
// Use fixed small sizes to keep the generated IR stable-ish.
extern "C" void kernel_gemv_int(const int *A, const int *x, int *y) {
  const int N = 4;
  for (int i = 0; i < N; ++i) {
    int acc = 0;
    for (int j = 0; j < N; ++j) {
      acc += A[i * N + j] * x[j];
    }
    y[i] = acc;
  }
}

int main() {
  static int A[16];
  static int x[4];
  static int y[4];
  kernel_gemv_int(A, x, y);
  return 0;
}


