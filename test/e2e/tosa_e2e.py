import subprocess
import os
import sys

# RUN: %python %s

# Define the TOSA input with embedded FileCheck comments
tosa_mlir = """// RUN: mlir-neura-opt %s --pass-pipeline='builtin.module(func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-elementwise-fusion),one-shot-bufferize{bufferize-function-boundaries=1},func.func(convert-linalg-to-affine-loops,convert-affine-to-taskflow))' | FileCheck %s

func.func @test_e2e(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %1 = tosa.mul %0, %0 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// CHECK-LABEL: taskflow.task
// CHECK: ^bb0(%[[ARG_A:.*]]: memref<16xf32>, %[[ARG_B:.*]]: memref<16xf32>, %[[ARG_OUT:.*]]: memref<16xf32>):
// CHECK-NEXT: affine.for %[[IV:.*]] = 0 to 16 {
// CHECK-NEXT:   %0 = affine.load %[[ARG_A]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT:   %1 = affine.load %[[ARG_B]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT:   %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:   %3 = arith.mulf %2, %2 : f32
// CHECK-NEXT:   affine.store %3, %[[ARG_OUT]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT: }
// CHECK-NEXT: "taskflow.yield"(%[[ARG_OUT]])
"""

def run_test():
    test_file = "tosa_e2e_temp.mlir"
    
    # Write the test case
    with open(test_file, "w") as f:
        f.write(tosa_mlir)
    
    neura_opt = "./tools/mlir-neura-opt/mlir-neura-opt"
    filecheck = "/home/x/shiran/llvm-project/build/bin/FileCheck"
    
    # Quote the pipeline argument for the shell
    pipeline = (
        "'builtin.module("
        "func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-fuse-elementwise-ops),"
        "one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},"
        "func.func(convert-linalg-to-affine-loops),"
        "convert-affine-to-taskflow"
        ")'"
    )
    
    try:
        # Robustly locate build directory relative to this script
        # Script is in <root>/test/e2e/tosa_e2e.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        build_dir = os.path.join(project_root, "build")
        
        neura_opt = os.path.join(build_dir, "tools/mlir-neura-opt/mlir-neura-opt")
        filecheck = "/home/x/shiran/llvm-project/build/bin/FileCheck" # Hardcoded env path as per previous usage
        
        # Check if tools exist
        if not os.path.exists(neura_opt):
            # Fallback if running in a different environment
            neura_opt = "mlir-neura-opt" # Try PATH
            
        test_file_path = os.path.join(script_dir, test_file)
        
        # Write test file to script dir
        with open(test_file_path, "w") as f:
            f.write(tosa_mlir)
            
        cmd = f"{neura_opt} --pass-pipeline={pipeline} {test_file_path} | {filecheck} {test_file_path}"
        
        # print(f"Executing: {cmd}")
        
        # Run without changing CWD, using absolute paths
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Test FAILED")
            print("Stderr:", result.stderr)
            print("Stdout:", result.stdout)
            sys.exit(1)
        else:
            print("Test PASSED")
            
    finally:
        if 'test_file_path' in locals() and os.path.exists(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    run_test()
