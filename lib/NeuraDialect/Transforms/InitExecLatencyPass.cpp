//===- InitExecLatencyPass.cpp - Initialize Execution Latency --------------===//
//
// This pass initializes execution latency information.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace mlir;

#define GEN_PASS_DEF_INITEXECLATENCY
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// Helper function to parse YAML scalar to integer
static bool parseYamlScalarInt(const llvm::yaml::Node *node, int &result) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return false;
  llvm::SmallString<64> value_string;
  llvm::StringRef value_ref = scalar->getValue(value_string);
  long long temp_value = 0;
  if (value_ref.getAsInteger(10, temp_value))
    return false;
  result = static_cast<int>(temp_value);
  return true;
}

// Helper function to parse YAML scalar to string
static bool parseYamlScalarString(const llvm::yaml::Node *node,
                                   std::string &result) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return false;
  llvm::SmallString<64> value_string;
  llvm::StringRef value_ref = scalar->getValue(value_string);
  result = value_ref.str();
  return true;
}

// Parse latency YAML file: expects a mapping of operation names to latency values
static bool parseLatencyYaml(const std::string &file_path,
                              std::map<std::string, int> &latency_map) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer_or_err =
      llvm::MemoryBuffer::getFile(file_path);
  if (!buffer_or_err) {
    llvm::errs() << "[InitExecLatencyPass] Failed to open latency specification file: "
                 << file_path << "\n";
    return false;
  }

  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(*buffer_or_err), llvm::SMLoc());
  llvm::yaml::Stream yaml_stream(
      sm.getMemoryBuffer(sm.getMainFileID())->getBuffer(), sm);

  llvm::yaml::Document &yaml_doc = *yaml_stream.begin();
  if (yaml_stream.failed()) {
    llvm::errs() << "[InitExecLatencyPass] YAML parse error in: " << file_path << "\n";
    return false;
  }

  auto *root = yaml_doc.getRoot();
  if (!root) {
    llvm::errs() << "[InitExecLatencyPass] Empty YAML document\n";
    return false;
  }

  auto *root_map = llvm::dyn_cast<llvm::yaml::MappingNode>(root);
  if (!root_map) {
    llvm::errs() << "[InitExecLatencyPass] YAML root is not a mapping\n";
    return false;
  }

  for (auto &key_value_pair : *root_map) {
    auto *key_node =
        llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node)
      continue;
    
    std::string op_name;
    if (!parseYamlScalarString(key_node, op_name))
      continue;

    int latency_value = 0;
    if (!parseYamlScalarInt(key_value_pair.getValue(), latency_value))
      continue;

    latency_map[op_name] = latency_value;
  }

  return true;
}

void SetLatency(Operation *op, std::map<std::string, int> &latency_map) {
    // Get operation name and look up latency
    std::string op_name = op->getName().getStringRef().str();
    if (op_name.compare("neura.fused_op") == 0) {
      op_name = op->getAttrOfType<StringAttr>("pattern_name").getValue().str();
    }
    op_name = op_name.substr(op_name.find_last_of(".") + 1); // remove neura. prefix if exists
    auto it = latency_map.find(op_name);
    if (it != latency_map.end()) {
        op->setAttr("latency", 
        IntegerAttr::get(IntegerType::get(op->getContext(), 32), it->second));
    }
    else {
        op->setAttr("latency", 
            IntegerAttr::get(IntegerType::get(op->getContext(), 32), 1)); 
    }
}

struct InitExecLatencyPass
    : public PassWrapper<InitExecLatencyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitExecLatencyPass)
  
  InitExecLatencyPass() = default;
  InitExecLatencyPass(const InitExecLatencyPass &pass)
      : PassWrapper<InitExecLatencyPass, OperationPass<ModuleOp>>(pass) {}
  
  StringRef getArgument() const override { return "init-exec-latency"; }
  StringRef getDescription() const override {
    return "Initialize execution latency information.";
  }
  
  void runOnOperation() override {

    ModuleOp module_op = getOperation();
    llvm::errs() << "[InitExecLatencyPass] Running init-exec-latency pass\n";
    // Get latency spec file from global function (set by command line)
    std::string latency_file = mlir::neura::getLatencySpecFile();
    if (latency_file.empty()) {
      latency_file = "latency_map.yaml"; // default file name
    }
    
    llvm::errs() << "[InitExecLatencyPass] Latency file: " << latency_file << "\n";
    // Builds a map of operation name to latency
    std::map<std::string, int> latency_map;
    if (!parseLatencyYaml(latency_file, latency_map)) {
      llvm::errs() << "[InitExecLatencyPass] Failed to parse latency specification file: " << latency_file << "\n";
      return;
    }

    // Apply latency values to operations
    module_op.walk([&](Operation *op) {
      if (!op->getRegions().empty()) {
        for (Region &region : op->getRegions()) {
          region.walk([&](Operation *inner_op) {
            // Skip operations inside fused_op regions
            if (inner_op->getParentOp() && isa<neura::FusedOp>(inner_op->getParentOp())) {
              return;
            }

            if (inner_op->getName().getStringRef().str() == "neura.data_mov" || inner_op->getName().getStringRef().str() == "neura.reserve") {
              return;
            }
            
            SetLatency(inner_op, latency_map);
          });
        }
      }
    });
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createInitExecLatencyPass() {
  return std::make_unique<InitExecLatencyPass>();
}
} // namespace mlir::neura
