import subprocess
import os
import sys

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
        # Assuming run from 'build' dir usually, but let's be robust
        cwd = os.getcwd()
        if not cwd.endswith("build"):
             # If running from root, try to use build tools
             if os.path.exists("build/tools/mlir-neura-opt/mlir-neura-opt"):
                 build_dir = "build"
             else:
                 # Fallback/Guess
                 build_dir = "." 
        else:
             build_dir = "."

        # Override for this specific environment structure mentioned in previous steps
        # The user's cwd seems to be project root. Tools are in ./build/tools/... or similar.
        # But previous successful runs used cwd=build_dir.
        # Let's hardcode the expectation that we run this script from project root,
        # and we invoke tools in 'build'.
        
        build_dir = os.path.join(os.getcwd(), "build")
        
        # Command must reference the temp file properly.
        # Since we write test_file in CWD (root), and switch to build_dir, check path.
        test_file_path = os.path.join(os.getcwd(), test_file)
        
        cmd = f"./tools/mlir-neura-opt/mlir-neura-opt --pass-pipeline={pipeline} {test_file_path} | {filecheck} {test_file_path}"
        
        print(f"Executing in {build_dir}:")
        print(cmd)
        
        result = subprocess.run(cmd, shell=True, cwd=build_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Test FAILED")
            print("Stderr:", result.stderr)
            print("Stdout:", result.stdout)
            sys.exit(1)
        else:
            print("Test PASSED")
            # print(result.stdout) # FileCheck output is usually empty on success (silent) or confusing.
            
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    run_test()
