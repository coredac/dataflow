#ifndef TASKFLOW_TYPES_TD
#define TASKFLOW_TYPES_TD

include "TaskflowDialect/TaskflowDialect.td"
include "mlir/IR/AttrTypeBase.td"

//----------------------------------------------------------------------
// Base Class for TaskFlow Types.
//----------------------------------------------------------------------
class TaskflowTypeBase<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TaskflowDialect, name, traits>{
  let mnemonic = typeMnemonic;
}
#endif //TASKFLOW_TYPES_TD