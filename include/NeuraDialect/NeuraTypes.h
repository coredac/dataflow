#ifndef NEURA_TYPES_H
#define NEURA_TYPES_H

#include "NeuraDialect/NeuraDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace neura {

namespace detail {
// Storage class for predicated value type.
struct PredicatedValueStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<Type, IntegerType>; // valueType and predicateType

  PredicatedValueStorage(Type valueType, IntegerType predicateType)
      : valueType(valueType), predicateType(predicateType) {}

  // Required storage class methods.
  bool operator==(const KeyTy &key) const {
    return key.first == valueType && key.second == predicateType;
  }

  static PredicatedValueStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    // Allocate the storage instance and construct it
    return new (allocator.allocate<PredicatedValueStorage>())
        PredicatedValueStorage(key.first, key.second);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  Type valueType;            // The type being predicated
  IntegerType predicateType; // The predicate type (usually i1)
};
} // namespace detail

class PredicatedValue
    : public mlir::Type::TypeBase<PredicatedValue, mlir::Type,
                                  detail::PredicatedValueStorage> {
public:
  using Base = mlir::Type::TypeBase<PredicatedValue, mlir::Type,
                                    detail::PredicatedValueStorage>;
  static constexpr llvm::StringLiteral name = "data";

  using Base::Base;

  // Static method to create a PredicatedValue instance.
  static PredicatedValue get(MLIRContext *context, Type valueType,
                             IntegerType predicateType) {
    return Base::get(context, valueType, predicateType);
  }

  // Overload verify that takes two separate parameters.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              Type valueType, IntegerType predicateType) {
    return verify(emitError, std::make_pair(valueType, predicateType));
  }

  // New overload verify that accepts the KeyTy as expected by MLIR
  static LogicalResult
  verify(function_ref<InFlightDiagnostic()> emitError,
         const detail::PredicatedValueStorage::KeyTy &key) {
    if (!key.second.isInteger(1))
      return emitError() << "predicate must be i1 type";
    return success();
  }

  Type getValueType() const { return getImpl()->valueType; }
  IntegerType getPredicateType() const { return getImpl()->predicateType; }

  static Type parse(AsmParser &parser);
  void print(AsmPrinter &printer) const;
};

} // namespace neura
} // namespace mlir

#endif // NEURA_TYPES_H