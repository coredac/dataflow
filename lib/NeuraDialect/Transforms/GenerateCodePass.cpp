#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_GenerateCode
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

struct GenerateCodePass
    : public PassWrapper<GenerateCodePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateCodePass)

  StringRef getArgument() const override { return "generate-code"; }
  StringRef getDescription() const override {
    return "Generates JSON code from mapped Neura IR.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::json::Array functions_array;

    for (auto func : module.getOps<func::FuncOp>()) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura") {
        continue;
      }

      llvm::json::Object func_obj;
      func_obj["name"] = func.getName().str();

      if (auto ii_attr = func->getAttrOfType<IntegerAttr>("CompiledII"))
        func_obj["CompiledII"] = ii_attr.getInt();
      if (auto recMII_attr = func->getAttrOfType<IntegerAttr>("RecMII"))
        func_obj["RecMII"] = recMII_attr.getInt();
      if (auto resMII_attr = func->getAttrOfType<IntegerAttr>("ResMII"))
        func_obj["ResMII"] = resMII_attr.getInt();

      llvm::json::Array op_array;

      func.walk([&](Operation *op) {
        if (isa<func::ReturnOp>(op))
          return;

        llvm::json::Object op_obj;
        op_obj["name"] = op->getName().getStringRef().str();

        // Result types.
        llvm::json::Array result_types;
        for (auto result : op->getResults()) {
        std::string type_str;
        llvm::raw_string_ostream os(type_str);
        result.getType().print(os);
        result_types.push_back(os.str());
        }
        op_obj["result_types"] = std::move(result_types);

        // Operands.
        llvm::json::Array operand_indices;
        for (Value operand : op->getOperands()) {
          if (auto defining_op = operand.getDefiningOp())
            operand_indices.push_back(defining_op->getName().getStringRef().str());
          else
            operand_indices.push_back("block_arg");
        }
        op_obj["operands"] = std::move(operand_indices);

        // Constants.
        if (auto const_op = mlir::dyn_cast<neura::ConstantOp>(op)) {
          auto val_attr = const_op.getValue();
          if (val_attr) {
            if (auto int_attr = mlir::dyn_cast<IntegerAttr>(val_attr)) {
              op_obj["constant_value"] = std::to_string(int_attr.getInt());
            } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(val_attr)) {
              op_obj["constant_value"] = std::to_string(float_attr.getValueAsDouble());
            }
          }
        }

        // Mapping locs.
        llvm::json::Array loc_array;
        if (auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
          for (Attribute attr : attr_array) {
            if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
              llvm::json::Object loc_obj;
              if (auto idAttr = mlir::dyn_cast<IntegerAttr>(loc.get("id")))
                loc_obj["id"] = idAttr.getInt();
              if (auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource")))
                loc_obj["resource"] = resource_attr.getValue().str();
              if (auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step")))
                loc_obj["time_step"] = timestep_attr.getInt();
              loc_array.push_back(std::move(loc_obj));
            }
          }
        }
        op_obj["mapping_locs"] = std::move(loc_array);

        op_array.push_back(std::move(op_obj));
      });

      func_obj["operations"] = std::move(op_array);
      functions_array.push_back(std::move(func_obj));
    }

    // Final JSON object.
    llvm::json::Object root;
    root["functions"] = std::move(functions_array);

    // llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    std::error_code ec;
    llvm::raw_fd_ostream json_out("generated-instructions.json", ec);
    if (ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.json' for writing: " + ec.message());
        return signalPassFailure();
    }
    json_out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura
