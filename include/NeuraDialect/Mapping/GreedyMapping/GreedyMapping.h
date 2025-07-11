#ifndef NEURA_GREEDY_MAPPING_H
#define NEURA_GREEDY_MAPPING_H

#include "NeuraDialect/Mapping/MappingStrategy.h"

namespace mlir {
namespace neura {

class GreedyMapping : public MappingStrategy {
public:
  bool map(std::vector<Operation*>& sorted_ops,
         const Architecture& architecture,
         MappingState& mapping_state) override;
         
  std::string getName() const override { return "greedy_mapping"; }
};

} // namespace neura
} // namespace mlir

#endif // NEURA_GREEDY_MAPPING_STRATEGY_H