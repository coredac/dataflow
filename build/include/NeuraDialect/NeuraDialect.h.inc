/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Neura.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace neura {

class NeuraDialect : public ::mlir::Dialect {
  explicit NeuraDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~NeuraDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("neura");
  }

  // Type parsing/printing
  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &printer) const override;

  // Attribute parsing/printing
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;
};
} // namespace neura
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::neura::NeuraDialect)
