#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @A : memref<1x4x16x64xf32> = uninitialized
  memref.global @C : memref<1x4x16x64xf32> = uninitialized
  func.func @_Z6node30v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+01 : f32
    %0 = llvm.mlir.undef : i32
    %1 = memref.get_global @C : memref<1x4x16x64xf32>
    %2 = memref.get_global @A : memref<1x4x16x64xf32>
    affine.for %arg0 = 0 to 4 {
      affine.for %arg1 = 0 to 16 {
        affine.for %arg2 = 0 to 64 step 2 {
          %3 = affine.load %2[0, %arg0, %arg1, %arg2] : memref<1x4x16x64xf32>
          %4 = arith.mulf %3, %cst : f32
          affine.store %4, %1[0, %arg0, %arg1, %arg2] : memref<1x4x16x64xf32>
          %5 = affine.apply #map(%arg2)
          %6 = affine.load %2[0, %arg0, %arg1, %5] : memref<1x4x16x64xf32>
          %7 = arith.mulf %6, %cst : f32
          affine.store %7, %1[0, %arg0, %arg1, %5] : memref<1x4x16x64xf32>
        }
      }
    }
    return %0 : i32
  }
}

