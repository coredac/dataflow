/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Neura.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::neura::NeuraDialect)
namespace mlir {
namespace neura {

NeuraDialect::NeuraDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<NeuraDialect>())
    
     {
  
  initialize();
}

NeuraDialect::~NeuraDialect() = default;

} // namespace neura
} // namespace mlir
