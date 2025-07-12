/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::neura::PredicatedValueType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::neura::PredicatedValueType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::neura::PredicatedValueType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::neura::PredicatedValueType>([&](auto t) {
      printer << ::mlir::neura::PredicatedValueType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace neura {
namespace detail {
struct PredicatedValueTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, IntegerType>;
  PredicatedValueTypeStorage(Type valueType, IntegerType predicateType) : valueType(std::move(valueType)), predicateType(std::move(predicateType)) {}

  KeyTy getAsKey() const {
    return KeyTy(valueType, predicateType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (valueType == std::get<0>(tblgenKey)) && (predicateType == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static PredicatedValueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto valueType = std::move(std::get<0>(tblgenKey));
    auto predicateType = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<PredicatedValueTypeStorage>()) PredicatedValueTypeStorage(std::move(valueType), std::move(predicateType));
  }

  Type valueType;
  IntegerType predicateType;
};
} // namespace detail
PredicatedValueType PredicatedValueType::get(::mlir::MLIRContext *context, Type valueType, IntegerType predicateType) {
  return Base::get(context, std::move(valueType), std::move(predicateType));
}

PredicatedValueType PredicatedValueType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type valueType, IntegerType predicateType) {
  return Base::getChecked(emitError, context, valueType, predicateType);
}

::llvm::LogicalResult PredicatedValueType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type valueType, IntegerType predicateType) {
  if (::mlir::failed(verify(emitError, valueType, predicateType)))
    return ::mlir::failure();
  return ::mlir::success();
}

Type PredicatedValueType::getValueType() const {
  return getImpl()->valueType;
}

IntegerType PredicatedValueType::getPredicateType() const {
  return getImpl()->predicateType;
}

} // namespace neura
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::neura::PredicatedValueType)

#endif  // GET_TYPEDEF_CLASSES

