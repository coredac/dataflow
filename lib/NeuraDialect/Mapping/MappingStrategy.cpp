#include "NeuraDialect/Mapping/MappingStrategy.h"
#include "NeuraDialect/Mapping/GreedyMapping/GreedyMapping.h"
#include <memory>

namespace mlir {
namespace neura {

std::unique_ptr<MappingStrategy> createMappingStrategy(const std::string& name) {
  if (name == "greedy_mapping") {
    return std::make_unique<GreedyMapping>();
  } else {
    llvm::errs() << "Unsupported mapping strategy: " << name;
    return nullptr;
  }
}

} // namespace neura
} // namespace mlir