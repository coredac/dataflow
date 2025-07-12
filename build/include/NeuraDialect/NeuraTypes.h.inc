/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace neura {
class PredicatedValueType;
namespace detail {
struct PredicatedValueTypeStorage;
} // namespace detail
class PredicatedValueType : public ::mlir::Type::TypeBase<PredicatedValueType, ::mlir::Type, detail::PredicatedValueTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "neura.data";
  static constexpr ::llvm::StringLiteral dialectName = "neura";
  using Base::getChecked;
  static PredicatedValueType get(::mlir::MLIRContext *context, Type valueType, IntegerType predicateType);
  static PredicatedValueType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type valueType, IntegerType predicateType);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type valueType, IntegerType predicateType);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type valueType, IntegerType predicateType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getValueType() const;
  IntegerType getPredicateType() const;
};
} // namespace neura
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::neura::PredicatedValueType)

#endif  // GET_TYPEDEF_CLASSES

