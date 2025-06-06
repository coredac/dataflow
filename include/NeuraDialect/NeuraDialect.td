// NeuraDialect.td - TableGen description of the dialect.
#ifndef NEURA_DIALECT_TD
#define NEURA_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// First define the dialect
def NeuraDialect : Dialect {
  let name = "neura";
  let cppNamespace = "::mlir::neura";
  
  let summary = "A dialect for the Neura compiler infrastructure.";
  let description = [{
    This dialect is used for representing Neura operations and types.
  }];
  
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;

  let extraClassDeclaration = [{
  // Type parsing/printing
  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &printer) const override;

  // Attribute parsing/printing
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;
}];
}

#endif // NEURA_DIALECT_TD