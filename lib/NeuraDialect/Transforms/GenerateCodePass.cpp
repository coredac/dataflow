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

    llvm::json::Array functionsArray;

    for (auto func : module.getOps<func::FuncOp>()) {
      auto accelAttr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accelAttr || accelAttr.getValue() != "neura")
        continue;

      llvm::json::Object funcObj;
      funcObj["name"] = func.getName().str();

      if (auto iiAttr = func->getAttrOfType<IntegerAttr>("CompiledII"))
        funcObj["CompiledII"] = iiAttr.getInt();
      if (auto recAttr = func->getAttrOfType<IntegerAttr>("RecMII"))
        funcObj["RecMII"] = recAttr.getInt();
      if (auto resAttr = func->getAttrOfType<IntegerAttr>("ResMII"))
        funcObj["ResMII"] = resAttr.getInt();

      llvm::json::Array opArray;

      func.walk([&](Operation *op) {
        if (isa<func::ReturnOp>(op))
          return;

        llvm::json::Object opObj;
        opObj["name"] = op->getName().getStringRef().str();

        // Result types
        llvm::json::Array resultTypes;
        for (auto result : op->getResults()) {
        std::string typeStr;
        llvm::raw_string_ostream os(typeStr);
        result.getType().print(os);
        resultTypes.push_back(os.str());
        }
        opObj["result_types"] = std::move(resultTypes);

        // Operands
        llvm::json::Array operandIndices;
        for (Value operand : op->getOperands()) {
          if (auto definingOp = operand.getDefiningOp())
            operandIndices.push_back(definingOp->getName().getStringRef().str());
          else
            operandIndices.push_back("block_arg");
        }
        opObj["operands"] = std::move(operandIndices);

        // Constants
        if (auto constOp = mlir::dyn_cast<neura::ConstantOp>(op)) {
          auto valAttr = constOp.getValue();
          if (valAttr) {
            if (auto intAttr = mlir::dyn_cast<IntegerAttr>(valAttr)) {
              opObj["constant_value"] = std::to_string(intAttr.getInt());
            } else if (auto floatAttr = mlir::dyn_cast<FloatAttr>(valAttr)) {
              opObj["constant_value"] = std::to_string(floatAttr.getValueAsDouble());
            }
          }
        }

        // Mapping locs
        llvm::json::Array locArray;
        if (auto attrArray = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
          for (Attribute attr : attrArray) {
            if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
              llvm::json::Object locObj;
              if (auto idAttr = mlir::dyn_cast<IntegerAttr>(loc.get("id")))
                locObj["id"] = idAttr.getInt();
              if (auto resAttr = mlir::dyn_cast<StringAttr>(loc.get("resource")))
                locObj["resource"] = resAttr.getValue().str();
              if (auto tsAttr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step")))
                locObj["time_step"] = tsAttr.getInt();
              locArray.push_back(std::move(locObj));
            }
          }
        }
        opObj["mapping_locs"] = std::move(locArray);

        opArray.push_back(std::move(opObj));
      });

      funcObj["operations"] = std::move(opArray);
      functionsArray.push_back(std::move(funcObj));
    }

    // Final JSON object
    llvm::json::Object root;
    root["functions"] = std::move(functionsArray);

    // llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    std::error_code ec;
    llvm::raw_fd_ostream jsonOut("generated-instructions.json", ec);
    if (ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.json' for writing: " + ec.message());
        return signalPassFailure();
    }
    jsonOut << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura
