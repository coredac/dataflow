import os
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
test_path = os.path.join(PROJECT_ROOT, "test/mapping_quality/branch_for.mlir")

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)

cmd_ir = (
    f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt {test_path} "
    "--assign-accelerator --lower-llvm-to-neura --promote-func-arg-to-const --fold-constant --canonicalize-live-in "
    "--leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant "
    "--insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=simple\" "
    "--architecture-spec=test/arch_spec/architecture.yaml 2>/dev/null"
)
res = run_cmd(cmd_ir)
mapping_header = ""
for l in res.stdout.splitlines():
    if "func.func @loop_test" in l:
        mapping_header = l.strip()
        break

run_cmd(cmd_ir + " --generate-code")

yaml_lines = []
if os.path.exists("tmp-generated-instructions.yaml"):
    with open("tmp-generated-instructions.yaml", 'r') as f:
        yaml_lines = [l.strip("\n") for l in f.readlines()[:25]]

asm_lines = []
if os.path.exists("tmp-generated-instructions.asm"):
    with open("tmp-generated-instructions.asm", 'r') as f:
        asm_lines = [l.strip("\n") for l in f.readlines()[:50]]

# Standard parts
prefix_runs = [
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --leverage-predicated-value \\\n",
    "// RUN:   | FileCheck %s\n",
    "\n",
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --promote-func-arg-to-const \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --canonicalize-live-in \\\n",
    "// RUN:   | FileCheck %s -check-prefix=CANONICALIZE\n",
    "\n",
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --promote-func-arg-to-const \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --canonicalize-live-in \\\n",
    "// RUN:   --leverage-predicated-value \\\n",
    "// RUN:   --transform-ctrl-to-data-flow \\\n",
    "// RUN:   | FileCheck %s -check-prefix=CTRL2DATA\n",
    "\n",
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --promote-func-arg-to-const \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --canonicalize-live-in \\\n",
    "// RUN:   --leverage-predicated-value \\\n",
    "// RUN:   --transform-ctrl-to-data-flow \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   | FileCheck %s -check-prefix=FUSE\n",
    "\n",
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --promote-func-arg-to-const \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --canonicalize-live-in \\\n",
    "// RUN:   --leverage-predicated-value \\\n",
    "// RUN:   --transform-ctrl-to-data-flow \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --insert-data-mov \\\n",
    "// RUN:   | FileCheck %s -check-prefix=MOV\n",
    "\n",
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --promote-func-arg-to-const \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --canonicalize-live-in \\\n",
    "// RUN:   --leverage-predicated-value \\\n",
    "// RUN:   --transform-ctrl-to-data-flow \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --insert-data-mov \\\n",
    "// RUN:   --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=simple\" \\\n",
    "// RUN:   --architecture-spec=../arch_spec/architecture.yaml \\\n",
    "// RUN:   | FileCheck %s -check-prefix=MAPPING\n",
    "\n",
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --promote-func-arg-to-const \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --canonicalize-live-in \\\n",
    "// RUN:   --leverage-predicated-value \\\n",
    "// RUN:   --transform-ctrl-to-data-flow \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --insert-data-mov \\\n",
    "// RUN:   --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=simple\" \\\n",
    "// RUN:   --architecture-spec=../arch_spec/architecture.yaml \\\n",
    "// RUN:   --generate-code\n",
    "// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml -check-prefix=YAML\n",
    "// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM\n"
]

func_code = """
func.func @loop_test() -> f32 {
  %n = llvm.mlir.constant(10 : i64) : i64
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c1f = llvm.mlir.constant(3.0 : f32) : f32
  %acc_init = llvm.mlir.constant(0.0 : f32) : f32

  llvm.br ^bb1(%c0, %acc_init : i64, f32)

^bb1(%i: i64, %acc: f32):  // loop body + check + increment
  %next_acc = llvm.fadd %acc, %c1f : f32
  %i_next = llvm.add %i, %c1 : i64
  %cmp = llvm.icmp "slt" %i_next, %n : i64
  llvm.cond_br %cmp, ^bb1(%i_next, %next_acc : i64, f32), ^exit(%next_acc : f32)

^exit(%result: f32):
  return %result : f32
}
"""

check_sects = """
// CHECK:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CHECK-NEXT:   %0 = "neura.constant"() <{value = 10 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %1 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %2 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %3 = "neura.constant"() <{value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %4 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   neura.br %1, %4 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1
// CHECK-NEXT: ^bb1(%5: !neura.data<i64, i1>, %6: !neura.data<f32, i1>):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:   %7 = "neura.fadd"(%6, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:   %8 = "neura.add"(%5, %2) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:   %9 = "neura.icmp"(%8, %0) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:   neura.cond_br %9 : !neura.data<i1, i1> then %8, %7 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1 else %7 : !neura.data<f32, i1> to ^bb2
// CHECK-NEXT: ^bb2(%10: !neura.data<f32, i1>):  // pred: ^bb1
// CHECK-NEXT:   "neura.return"(%10) : (!neura.data<f32, i1>) -> ()
// CHECK-NEXT: }

// CANONICALIZE:       func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CANONICALIZE-NEXT:     %0 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %1 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
// CANONICALIZE-NEXT:     neura.br %0, %1 : i64, f32 to ^bb1
// CANONICALIZE-NEXT:   ^bb1(%2: i64, %3: f32):  // 2 preds: ^bb0, ^bb1
// CANONICALIZE-NEXT:     %4 = "neura.fadd"(%3) {rhs_value = 3.000000e+00 : f32} : (f32) -> f32
// CANONICALIZE-NEXT:     %5 = "neura.add"(%2) {rhs_value = 1 : i64} : (i64) -> i64
// CANONICALIZE-NEXT:     %6 = "neura.icmp"(%5) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (i64) -> i1
// CANONICALIZE-NEXT:     neura.cond_br %6 : i1 then %5, %4 : i64, f32 to ^bb1 else %4 : f32 to ^bb2
// CANONICALIZE-NEXT:   ^bb2(%7: f32):  // pred: ^bb1
// CANONICALIZE-NEXT:     "neura.return"(%7) : (f32) -> ()
// CANONICALIZE-NEXT:   }

// CTRL2DATA:        func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %4 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %5 = neura.phi_start %3, %4 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %6 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = neura.phi_start %1, %6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.fadd"(%5) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.add"(%7) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = "neura.icmp"(%9) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %11 = neura.grant_predicate %9, %10 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %11 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.grant_predicate %8, %10 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %12 -> %4 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %13 = "neura.not"(%10) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %14 = neura.grant_predicate %8, %13 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     "neura.return"(%14) : (!neura.data<f32, i1>) -> ()
// CTRL2DATA-NEXT:   }


// FUSE:        func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// FUSE-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// FUSE-NEXT:     %2 = neura.reserve : !neura.data<f32, i1>
// FUSE-NEXT:     %3 = neura.phi_start %1, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     %4 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:     %5 = neura.phi_start %0, %4 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     %6 = "neura.fadd"(%3) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %7 = "neura.add"(%5) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %8 = "neura.icmp"(%7) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %9 = neura.grant_predicate %7, %8 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     neura.ctrl_mov %9 -> %4 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:     %10 = neura.grant_predicate %6, %8 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     neura.ctrl_mov %10 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
// FUSE-NEXT:     %11 = "neura.not"(%8) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %12 = neura.grant_predicate %6, %11 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     "neura.return"(%12) : (!neura.data<f32, i1>) -> ()
// FUSE-NEXT:   }

// MOV:        func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// MOV-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %2 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     %5 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %9 = \"neura.fadd\"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %10 = \"neura.data_mov\"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %11 = \"neura.add\"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %12 = \"neura.data_mov\"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %13 = \"neura.icmp\"(%12) <{cmpType = \"slt\"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %14 = \"neura.data_mov\"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %15 = \"neura.data_mov\"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %17 = \"neura.data_mov\"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %18 = \"neura.data_mov\"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %20 = \"neura.data_mov\"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %21 = \"neura.not\"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %22 = \"neura.data_mov\"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %23 = \"neura.data_mov\"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     %25 = \"neura.data_mov\"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     \"neura.return\"(%25) : (!neura.data<f32, i1>) -> ()
// MOV-NEXT:   }
"""

with open(test_path, 'w') as f:
    f.writelines(prefix_runs)
    f.write(func_code)
    f.write(check_sects)
    f.write("\n")
    f.write(f"// MAPPING:        {mapping_header}\n")
    f.write("\n")
    # YAML - Use CHECK for the first line of the block if it's not the very first line of the file
    first_yaml = True
    for l in yaml_lines:
        if not l.strip(): continue
        pref = "YAML" if (first_yaml or "column:" in l or "core_id:" in l or "array_config" in l) else "YAML-NEXT"
        f.write(f"// {pref}:      {l}\n")
        first_yaml = False
    f.write("\n")
    # ASM - Use CHECK (no NEXT) for every line starting with PE( or # Compiled
    first_asm = True
    for l in asm_lines:
        if not l.strip(): continue
        # Use CHECK for PE lines to jump over empty lines in output
        pref = "ASM" if (first_asm or l.startswith("PE(")) else "ASM-NEXT"
        f.write(f"// {pref}:      {l}\n")
        first_asm = False

