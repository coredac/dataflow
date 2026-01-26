// Keep the function name matching "kernel" so llvm-extract --rfunc=".*kernel.*" can find it.
extern "C" void kernel_axpy_int(int n, int a, const int *x, int *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// Provide a tiny main so clang emits a complete TU; tests extract only the kernel anyway.
int main() {
  const int N = 16;
  static int x[N];
  static int y[N];
  kernel_axpy_int(N, /*a=*/3, x, y);
  return 0;
}


