module {
  func.func @parallel_nested_example(%arg0: memref<16xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: f32) {
    %write_outputs = taskflow.task @Task_0 read_memrefs(%arg0 : memref<16xf32>) write_memrefs(%arg0 : memref<16xf32>) value_inputs(%arg4 : f32) [original_read_memrefs(%arg0 : memref<16xf32>), original_write_memrefs(%arg0 : memref<16xf32>)] : (memref<16xf32>, memref<16xf32>, f32) -> (memref<16xf32>) {
    ^bb0(%arg5: memref<16xf32>, %arg6: memref<16xf32>, %arg7: f32):
      affine.for %arg8 = 0 to 16 {
        %0 = affine.load %arg6[%arg8] : memref<16xf32>
        %1 = arith.mulf %0, %arg7 : f32
        affine.store %1, %arg6[%arg8] : memref<16xf32>
      }
      taskflow.yield writes(%arg6 : memref<16xf32>)
    }
    %write_outputs_0 = taskflow.task @Task_1 read_memrefs(%arg1, %arg2 : memref<8x8xf32>, memref<8x8xf32>) write_memrefs(%arg3 : memref<8x8xf32>) [original_read_memrefs(%arg1, %arg2 : memref<8x8xf32>, memref<8x8xf32>), original_write_memrefs(%arg3 : memref<8x8xf32>)] : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> (memref<8x8xf32>) {
    ^bb0(%arg5: memref<8x8xf32>, %arg6: memref<8x8xf32>, %arg7: memref<8x8xf32>):
      affine.for %arg8 = 0 to 8 {
        affine.for %arg9 = 0 to 8 {
          %0 = affine.load %arg5[%arg8, %arg9] : memref<8x8xf32>
          %1 = affine.load %arg6[%arg8, %arg9] : memref<8x8xf32>
          %2 = arith.mulf %0, %1 : f32
          affine.store %2, %arg7[%arg8, %arg9] : memref<8x8xf32>
        }
      }
      taskflow.yield writes(%arg7 : memref<8x8xf32>)
    }
    return
  }
}

