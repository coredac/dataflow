// Keep the function name matching "kernel" so llvm-extract --rfunc=".*kernel.*" can find it.
extern "C" void kernel_axpy_int(const int *x, int *y) {
  constexpr int N = 16;
  constexpr int A = 3;
  for (int i = 0; i < N; ++i) {
    y[i] = A * x[i] + y[i];
  }
}

// Provide a tiny main so clang emits a complete TU; tests extract only the kernel anyway.
int main() {
  constexpr int N = 16;
  static int x[N];
  static int y[N];
  kernel_axpy_int(x, y);
  return 0;
}


