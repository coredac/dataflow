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

//----------------------------------------------------------------------
// PacketType - Control conifguration packet type.
//----------------------------------------------------------------------
def TaskflowPacketType : TaskflowTypeBase<"Packet", "packet">{
  let summary = "Control packet carrying conifguration metadata for affine controller";

  let description = [{
    
  }];

  // Payload type carried by the packet.
  let parameters = (ins "::mlir::Type":$payloadType);

  let assemblyFormat = [{
    `<` $payloadType `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$payloadType),
      [{
        return $_get(payloadType.getContext(), payloadType);
      }]>
  ];
}
#endif //TASKFLOW_TYPES_TD