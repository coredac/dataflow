// RUN: clang++ -S -emit-llvm kernel.cpp -o %t-kernel.ll
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov %t-kernel.mlir \
// RUN:   | FileCheck %s --check-prefix=CHECK-LLVM2NEURA

// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-LLVM2NEURA-MAP

// CHECK-LLVM2NEURA: accelerator = "neura"
// CHECK-LLVM2NEURA: %25 = neura.alloca %24 : !neura.data<i32, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-LLVM2NEURA: %38 = neura.phi_start %37, %36 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CHECK-LLVM2NEURA: %188 = neura.sext %187 : !neura.data<i32, i1> -> !neura.data<i64, i1>
// CHECK-LLVM2NEURA: %207 = "neura.mul"(%205, %206) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>

// CHECK-LLVM2NEURA-MAP: Usage:	/bin/sh [GNU long option] [option] ...
// CHECK-LLVM2NEURA-MAP-NEXT: 	/bin/sh [GNU long option] [option] script-file ...
// CHECK-LLVM2NEURA-MAP-NEXT: GNU long options:
// CHECK-LLVM2NEURA-MAP-NEXT: 	--debug
// CHECK-LLVM2NEURA-MAP-NEXT: 	--debugger
// CHECK-LLVM2NEURA-MAP-NEXT: 	--dump-po-strings
// CHECK-LLVM2NEURA-MAP-NEXT: 	--dump-strings
// CHECK-LLVM2NEURA-MAP-NEXT: 	--help
// CHECK-LLVM2NEURA-MAP-NEXT: 	--init-file
// CHECK-LLVM2NEURA-MAP-NEXT: 	--login
// CHECK-LLVM2NEURA-MAP-NEXT: 	--noediting
// CHECK-LLVM2NEURA-MAP-NEXT: 	--noprofile
// CHECK-LLVM2NEURA-MAP-NEXT: 	--norc
// CHECK-LLVM2NEURA-MAP-NEXT: 	--posix
// CHECK-LLVM2NEURA-MAP-NEXT: 	--pretty-print
// CHECK-LLVM2NEURA-MAP-NEXT: 	--rcfile
// CHECK-LLVM2NEURA-MAP-NEXT: 	--restricted
// CHECK-LLVM2NEURA-MAP-NEXT: 	--verbose
// CHECK-LLVM2NEURA-MAP-NEXT: 	--version
// CHECK-LLVM2NEURA-MAP-NEXT: Shell options:
// CHECK-LLVM2NEURA-MAP-NEXT: 	-ilrsD or -c command or -O shopt_option		(invocation only)
// CHECK-LLVM2NEURA-MAP-NEXT: 	-abefhkmnptuvxBCHP or -o option
