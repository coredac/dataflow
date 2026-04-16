#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace taskflow {
namespace attr {
// Attribute keys and values related to each task.
constexpr llvm::StringLiteral kTaskInfo = "task_info";

// Attribute keys on taskflow.task operations produced by the
// TaskDivisibilityAnalysisPass.
constexpr llvm::StringLiteral kDivisibility = "divisibility";
constexpr llvm::StringLiteral kParallelDims = "parallel_dims";
constexpr llvm::StringLiteral kParallelSpace = "parallel_space";

namespace val {
constexpr llvm::StringLiteral kDivisible = "divisible";
constexpr llvm::StringLiteral kAtomic = "atomic";
} // namespace val
} // namespace attr
} // namespace taskflow
} // namespace mlir