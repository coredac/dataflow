import subprocess
import os
import sys
import argparse

# RUN: %python %s

# Defines the TOSA input with embedded FileCheck comments.
tosa_mlir = """
func.func @test_e2e(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %1 = tosa.mul %0, %0 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_e2e
// CHECK: %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK: %[[RES:.*]] = "taskflow.task"(%arg0, %arg1, %alloc)
// CHECK-SAME: task_name = "Task_0"
// CHECK-NEXT: ^bb0(%[[BA1:.*]]: memref<16xf32>, %[[BA2:.*]]: memref<16xf32>, %[[BA3:.*]]: memref<16xf32>):
// CHECK-NEXT:   affine.for %[[IV:.*]] = 0 to 16 {
// CHECK-NEXT:     %0 = affine.load %[[BA1]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT:     %1 = affine.load %[[BA2]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:     %3 = arith.mulf %2, %2 : f32
// CHECK-NEXT:     affine.store %3, %[[BA3]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "taskflow.yield"(%[[BA3]])
// CHECK: return %[[RES]] : memref<16xf32>
"""

def run_test():
    # Parses arguments for tool paths.
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir-neura-opt", help="Path to mlir-neura-opt")
    parser.add_argument("--filecheck", help="Path to FileCheck")
    args, unknown = parser.parse_known_args()

    test_file = "tosa_e2e_temp.mlir"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(script_dir, test_file)
    
    # Writes the test case to a temporary file.
    with open(test_file_path, "w") as f:
        f.write(tosa_mlir)
    
    # Determines the tool paths.
    neura_opt = args.mlir_neura_opt
    filecheck = args.filecheck

    # Fallback to defaults or environment if tools are not provided via arguments.
    if not neura_opt:
        project_root = os.path.dirname(os.path.dirname(script_dir))
        neura_opt = os.path.join(project_root, "build", "tools", "mlir-neura-opt", "mlir-neura-opt")
    if not filecheck:
        filecheck = "FileCheck"

    # Defines the lowering pipeline.
    pipeline = (
        "'builtin.module("
        "func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-fuse-elementwise-ops),"
        "one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},"
        "func.func(convert-linalg-to-affine-loops),"
        "convert-affine-to-taskflow"
        ")'"
    )
    
    try:
        # Constructs the command string.
        cmd = f"{neura_opt} --pass-pipeline={pipeline} {test_file_path} | {filecheck} {test_file_path}"
        
        # Runs the command and captures output.
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Test FAILED")
            print("Stderr:", result.stderr)
            print("Stdout:", result.stdout)
            sys.exit(1)
        else:
            print("Test PASSED")
            
    finally:
        # Cleans up the temporary test file.
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    run_test()
