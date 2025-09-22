// RUN: clang++ -S -emit-llvm kernel2.cpp -o kernel2.ll
// RUN: mlir-translate --import-llvm kernel2.ll -o kernel2.mlir
// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" \
// RUN:   --generate-code kernel2.mlir | FileCheck %s --check-prefix=CHECK-LLVM2NEURA

// CHECK-LLVM2NEURA: %25 = neura.alloca %24 : !neura.data<i32, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-LLVM2NEURA: %38 = "neura.phi"(%36, %37) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-LLVM2NEURA: %175 = neura.sext %174 : !neura.data<i32, i1> -> !neura.data<i64, i1>
// CHECK-LLVM2NEURA: %194 = "neura.mul"(%192, %193) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>