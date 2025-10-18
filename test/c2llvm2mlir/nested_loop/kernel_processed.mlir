module attributes {dlti.dl_spec = #dlti.dl_spec<f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external @input(dense<1> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
  llvm.mlir.global external @output(dense<0> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
  llvm.mlir.global external @coefficients(dense<[25, 150, 375, -225, 50, 75, -300, 125, 25, 150, 375, -225, 50, 75, -300, 125, 25, 150, 375, -225, 50, 75, -300, 125, 25, 150, 375, -225, 50, 75, -300, 125]> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
  llvm.mlir.global private unnamed_addr constant @".str"("output: %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main() -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, optimize_none, passthrough = ["mustprogress", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    %1 = llvm.mlir.addressof @coefficients : !llvm.ptr
    %2 = llvm.mlir.addressof @output : !llvm.ptr
    %3 = llvm.mlir.addressof @input : !llvm.ptr
    %4 = "neura.constant"() <{value = 1 : i32}> : () -> i32
    %5 = "neura.constant"() <{value = 0 : i32}> : () -> i32
    %6 = "neura.data_mov"(%4) : (i32) -> i32
    %7 = neura.alloca %6 : i32 -> !llvm.ptr
    %8 = "neura.data_mov"(%5) : (i32) -> i32
    %9 = "neura.data_mov"(%7) : (!llvm.ptr) -> !llvm.ptr
    "neura.store"(%8, %9) : (i32, !llvm.ptr) -> ()
    %10 = func.call @_Z6kernelPiS_S_(%3, %2, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.void
    %11 = "neura.data_mov"(%2) : (!llvm.ptr) -> !llvm.ptr
    %12 = "neura.load"(%11) : (!llvm.ptr) -> i32
    %13 = llvm.call @printf(%0, %12) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %14 = "neura.data_mov"(%5) : (i32) -> i32
    "neura.return"(%14) : (i32) -> ()
  }
  func.func @_Z6kernelPiS_S_(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", frame_pointer = #llvm.framePointerKind<all>, linkage = #llvm.linkage<external>, no_inline, no_unwind, optimize_none, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 0 : i64, visibility_ = 0 : i64} {
    %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
    %1 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
    %2 = "neura.constant"() <{value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
    %3 = "neura.grant_once"() <{constant_value = 1 : i32}> : () -> !neura.data<i32, i1>
    %4 = "neura.constant"() <{value = 1 : i32}> : () -> !neura.data<i32, i1>
    %5 = "neura.grant_once"() <{constant_value = 0 : i32}> : () -> !neura.data<i32, i1>
    %6 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
    %7 = "neura.grant_once"() <{constant_value = 32 : i32}> : () -> !neura.data<i32, i1>
    %8 = "neura.data_mov"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %9 = neura.alloca %8 : !neura.data<i32, i1> -> !neura.data<!llvm.ptr, i1>
    %10 = "neura.data_mov"(%9) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %11 = "neura.grant_once"(%10) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %12 = "neura.data_mov"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %13 = neura.alloca %12 : !neura.data<i32, i1> -> !neura.data<!llvm.ptr, i1>
    %14 = "neura.data_mov"(%13) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %15 = "neura.grant_once"(%14) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %16 = "neura.data_mov"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %17 = neura.alloca %16 : !neura.data<i32, i1> -> !neura.data<!llvm.ptr, i1>
    %18 = "neura.data_mov"(%17) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %19 = "neura.grant_once"(%18) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %20 = "neura.data_mov"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %21 = neura.alloca %20 : !neura.data<i32, i1> -> !neura.data<!llvm.ptr, i1>
    %22 = "neura.data_mov"(%21) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %23 = "neura.grant_once"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %24 = "neura.data_mov"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %25 = neura.alloca %24 : !neura.data<i32, i1> -> !neura.data<!llvm.ptr, i1>
    %26 = "neura.data_mov"(%25) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %27 = "neura.grant_once"(%26) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %28 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %29 = "neura.data_mov"(%9) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%28, %29) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %30 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %31 = "neura.data_mov"(%13) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%30, %31) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %32 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %33 = "neura.data_mov"(%17) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%32, %33) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %34 = "neura.data_mov"(%6) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %35 = "neura.data_mov"(%21) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%34, %35) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %36 = neura.reserve : !neura.data<i32, i1>
    %37 = "neura.data_mov"(%3) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %38 = "neura.phi"(%36, %37) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %39 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %40 = "neura.data_mov"(%15) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %41 = "neura.phi"(%39, %40) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %42 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %43 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %44 = "neura.phi"(%42, %43) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %45 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %46 = "neura.data_mov"(%11) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %47 = "neura.phi"(%45, %46) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %48 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %49 = "neura.data_mov"(%27) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %50 = "neura.phi"(%48, %49) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %51 = neura.reserve : !neura.data<i32, i1>
    %52 = "neura.data_mov"(%5) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %53 = "neura.phi"(%51, %52) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %54 = neura.reserve : !neura.data<i32, i1>
    %55 = "neura.data_mov"(%7) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %56 = "neura.phi"(%54, %55) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %57 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %58 = "neura.data_mov"(%23) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %59 = "neura.phi"(%57, %58) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %60 = "neura.data_mov"(%59) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %61 = "neura.load"(%60) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %62 = "neura.data_mov"(%61) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %63 = "neura.data_mov"(%56) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %64 = "neura.icmp"(%62, %63) <{cmpType = "slt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
    %65 = "neura.data_mov"(%53) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %66 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %67 = neura.grant_predicate %65, %66 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %68 = "neura.data_mov"(%50) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %69 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %70 = neura.grant_predicate %68, %69 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %71 = "neura.data_mov"(%56) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %72 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %73 = neura.grant_predicate %71, %72 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %74 = "neura.data_mov"(%47) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %75 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %76 = neura.grant_predicate %74, %75 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %77 = "neura.data_mov"(%59) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %78 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %79 = neura.grant_predicate %77, %78 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %80 = "neura.data_mov"(%44) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %81 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %82 = neura.grant_predicate %80, %81 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %83 = "neura.data_mov"(%41) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %84 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %85 = neura.grant_predicate %83, %84 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %86 = "neura.data_mov"(%38) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %87 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %88 = neura.grant_predicate %86, %87 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %89 = "neura.data_mov"(%67) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %90 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%89, %90) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %91 = neura.reserve : !neura.data<i32, i1>
    %92 = "neura.data_mov"(%67) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %93 = "neura.phi"(%91, %92) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %94 = neura.reserve : !neura.data<i32, i1>
    %95 = "neura.data_mov"(%88) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %96 = "neura.phi"(%94, %95) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %97 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %98 = "neura.data_mov"(%85) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %99 = "neura.phi"(%97, %98) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %100 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %101 = "neura.data_mov"(%82) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %102 = "neura.phi"(%100, %101) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %103 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %104 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %105 = "neura.phi"(%103, %104) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %106 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %107 = "neura.data_mov"(%76) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %108 = "neura.phi"(%106, %107) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %109 = neura.reserve : !neura.data<i32, i1>
    %110 = "neura.data_mov"(%73) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %111 = "neura.phi"(%109, %110) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %112 = neura.reserve : !neura.data<!llvm.ptr, i1>
    %113 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %114 = "neura.phi"(%112, %113) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %115 = "neura.data_mov"(%114) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %116 = "neura.load"(%115) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %117 = "neura.data_mov"(%116) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %118 = "neura.data_mov"(%111) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %119 = "neura.icmp"(%117, %118) <{cmpType = "slt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
    %120 = "neura.data_mov"(%108) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %121 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %122 = neura.grant_predicate %120, %121 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %123 = "neura.data_mov"(%105) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %124 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %125 = neura.grant_predicate %123, %124 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %126 = "neura.data_mov"(%102) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %127 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %128 = neura.grant_predicate %126, %127 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %129 = "neura.data_mov"(%99) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %130 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %131 = neura.grant_predicate %129, %130 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %132 = "neura.data_mov"(%114) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %133 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %134 = neura.grant_predicate %132, %133 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %135 = "neura.data_mov"(%96) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %136 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %137 = neura.grant_predicate %135, %136 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %138 = "neura.data_mov"(%111) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %139 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %140 = neura.grant_predicate %138, %139 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %141 = "neura.data_mov"(%93) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %142 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %143 = neura.grant_predicate %141, %142 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %144 = "neura.data_mov"(%119) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %145 = "neura.not"(%144) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %146 = "neura.data_mov"(%105) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %147 = "neura.data_mov"(%145) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %148 = neura.grant_predicate %146, %147 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %149 = "neura.data_mov"(%96) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %150 = "neura.data_mov"(%145) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %151 = neura.grant_predicate %149, %150 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %152 = "neura.data_mov"(%111) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %153 = "neura.data_mov"(%145) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %154 = neura.grant_predicate %152, %153 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %155 = "neura.data_mov"(%93) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %156 = "neura.data_mov"(%145) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %157 = neura.grant_predicate %155, %156 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
    %158 = "neura.data_mov"(%114) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %159 = "neura.data_mov"(%145) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %160 = neura.grant_predicate %158, %159 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %161 = "neura.data_mov"(%108) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %162 = "neura.data_mov"(%145) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %163 = neura.grant_predicate %161, %162 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %164 = "neura.data_mov"(%102) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %165 = "neura.data_mov"(%145) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %166 = neura.grant_predicate %164, %165 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %167 = "neura.data_mov"(%99) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %168 = "neura.data_mov"(%145) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %169 = neura.grant_predicate %167, %168 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %170 = "neura.data_mov"(%122) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %171 = "neura.load"(%170) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %172 = "neura.data_mov"(%125) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %173 = "neura.load"(%172) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %174 = "neura.data_mov"(%173) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %175 = neura.sext %174 : !neura.data<i32, i1> -> !neura.data<i64, i1>
    %176 = "neura.data_mov"(%171) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %177 = "neura.data_mov"(%175) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %178 = "neura.gep"(%176, %177) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %179 = "neura.data_mov"(%178) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %180 = "neura.load"(%179) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %181 = "neura.data_mov"(%128) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %182 = "neura.load"(%181) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %183 = "neura.data_mov"(%125) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %184 = "neura.load"(%183) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %185 = "neura.data_mov"(%184) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %186 = neura.sext %185 : !neura.data<i32, i1> -> !neura.data<i64, i1>
    %187 = "neura.data_mov"(%182) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %188 = "neura.data_mov"(%186) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %189 = "neura.gep"(%187, %188) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %190 = "neura.data_mov"(%189) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %191 = "neura.load"(%190) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %192 = "neura.data_mov"(%180) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %193 = "neura.data_mov"(%191) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %194 = "neura.mul"(%192, %193) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %195 = "neura.data_mov"(%131) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %196 = "neura.load"(%195) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %197 = "neura.data_mov"(%134) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %198 = "neura.load"(%197) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %199 = "neura.data_mov"(%198) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %200 = neura.sext %199 : !neura.data<i32, i1> -> !neura.data<i64, i1>
    %201 = "neura.data_mov"(%196) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %202 = "neura.data_mov"(%200) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %203 = "neura.gep"(%201, %202) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %204 = "neura.data_mov"(%203) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %205 = "neura.load"(%204) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %206 = "neura.data_mov"(%205) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %207 = "neura.data_mov"(%194) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %208 = "neura.add"(%206, %207) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %209 = "neura.data_mov"(%208) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %210 = "neura.data_mov"(%203) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%209, %210) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %211 = "neura.data_mov"(%134) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %212 = "neura.load"(%211) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %213 = "neura.data_mov"(%212) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %214 = "neura.data_mov"(%137) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %215 = "neura.add"(%213, %214) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %216 = "neura.data_mov"(%215) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %217 = "neura.data_mov"(%134) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%216, %217) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    neura.ctrl_mov %134 -> %112 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %140 -> %109 : !neura.data<i32, i1> !neura.data<i32, i1>
    neura.ctrl_mov %122 -> %106 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %125 -> %103 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %128 -> %100 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %131 -> %97 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %137 -> %94 : !neura.data<i32, i1> !neura.data<i32, i1>
    neura.ctrl_mov %143 -> %91 : !neura.data<i32, i1> !neura.data<i32, i1>
    %218 = "neura.data_mov"(%148) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %219 = "neura.load"(%218) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %220 = "neura.data_mov"(%219) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %221 = "neura.data_mov"(%151) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %222 = "neura.add"(%220, %221) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %223 = "neura.data_mov"(%222) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %224 = "neura.data_mov"(%148) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%223, %224) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    neura.ctrl_mov %148 -> %57 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %154 -> %54 : !neura.data<i32, i1> !neura.data<i32, i1>
    neura.ctrl_mov %157 -> %51 : !neura.data<i32, i1> !neura.data<i32, i1>
    neura.ctrl_mov %160 -> %48 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %163 -> %45 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %166 -> %42 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %169 -> %39 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %151 -> %36 : !neura.data<i32, i1> !neura.data<i32, i1>
    "neura.return"() : () -> ()
  }
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
}

