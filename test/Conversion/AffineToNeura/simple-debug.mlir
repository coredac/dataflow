// Simple test to debug the issue
func.func @simple_loop(%A: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %v = affine.load %A[%i] : memref<10xf32>
  }
  return
}
