//===- Allocation.h - Abstract base class for CGRA task allocation --------===//
//
// Defines the abstract Allocation interface for mapping Taskflow tasks onto a
// 2D multi-CGRA grid.  Concrete strategies (e.g. ProximityAllocation) derive
// from this class and override runAllocation().
//
// Modelled after include/NeuraDialect/Mapping/Mapping.h.
//
//===----------------------------------------------------------------------===//

#ifndef TASKFLOW_ALLOCATION_H
#define TASKFLOW_ALLOCATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <string>

namespace mlir {
namespace taskflow {

//===----------------------------------------------------------------------===//
// Allocation — abstract base class
//===----------------------------------------------------------------------===//

/// Abstract base class for different CGRA task-allocation strategies.
///
/// Subclasses implement runAllocation() to map every taskflow.task operation
/// inside `func` onto the physical 2D multi-CGRA grid.  The pass delegates to
/// whichever concrete strategy is installed, making it straightforward to swap
/// in alternative algorithms (e.g. ILP-based, simulated-annealing, etc.)
/// without touching the pass infrastructure.
class Allocation {
public:
  virtual ~Allocation() = default;

  /// Runs the allocation strategy on `func`, annotating each
  /// taskflow.task op with a `task_allocation_info` attribute that records
  /// the assigned CGRA positions and SRAM locations.
  ///
  /// Returns true on success, false if no valid placement could be found
  /// (e.g. the grid is too full).
  virtual bool runAllocation(mlir::func::FuncOp func) = 0;

  /// Returns a human-readable name for this strategy (used in log messages).
  virtual std::string getName() const = 0;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_ALLOCATION_H
