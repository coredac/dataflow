namespace {
constexpr int kN = 16;
constexpr int kA = 3;
}  // namespace

// Keep the function name matching "kernel" so llvm-extract --rfunc=".*kernel.*" can find it.
extern "C" void kernel_axpy_int(const int *x, int *y) {
  for (int i = 0; i < kN; ++i) {
    y[i] = kA * x[i] + y[i];
  }
}

// Provide a tiny main so clang emits a complete TU; tests extract only the kernel anyway.
int main() {
  static int x[kN];
  static int y[kN];
  kernel_axpy_int(x, y);
  return 0;
}


