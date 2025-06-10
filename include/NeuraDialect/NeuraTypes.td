#ifndef NEURA_TYPES_TD
#define NEURA_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "NeuraDialect.td"

// Define the Neura dialect types
// class Neura_Type<string name, list<Trait> traits = []> :
//    TypeDef<Neura_Dialect, name, traits>;
class Neura_Type<string name, string typeMnemonic> : TypeDef<NeuraDialect, name> {
  let mnemonic = typeMnemonic;  // Changed parameter name to avoid self-assignment
}

// Predicated value type - a value with an optional predicate
def Neura_PredicatedValue : Neura_Type<"PredicatedValue", "data"> {
  let summary = "A value with an optional predicate";
  let description = [{
    Represents a value that may be conditional based on a predicate.
    Contains:
    - A base value type (integer, float, or vector)
    - An i1 predicate indicating validity
    
    Examples:
      !neura.data<f32>    // Predicated float
      !neura.data<i32>    // Predicated integer
      !neura.data<vector> // Predicated vector
  }];

  let parameters = (ins 
    "Type":$valueType,
    "IntegerType":$predicateType
  );

  // Verify predicate is i1
  let genVerifyDecl = 1;
  
  // Fix: Update assembly format syntax with proper semicolon
  // let assemblyFormat = "<$valueType,$predicateType>";
  let hasCustomAssemblyFormat = 1;
  // let mnemonic = "data";
  // let cppNamespace = "::mlir::neura";
}

#endif // NEURA_TYPES_TD