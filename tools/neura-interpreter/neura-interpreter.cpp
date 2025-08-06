#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"

#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <cassert>

using namespace mlir;

static llvm::DenseMap<Value, llvm::SmallPtrSet<Operation*, 4>> valueUsers;
static llvm::SmallVector<Operation*, 16> worklist;
static llvm::DenseMap<Operation*, bool> inWorklist;

static bool verbose = false;
static bool dataflow = false;

inline void setDataflowMode(bool v) {
  dataflow = v;
}

inline bool isDataflowMode() {
  return dataflow;
}

inline void setVerboseMode(bool v) {
  verbose = v;
}

inline bool isVerboseMode() {
  return verbose;
}

void buildDependencyGraph(ModuleOp module) {
  valueUsers.clear();

  module.walk([&](Operation* op) {
    if (isa<neura::ReturnOp>(op) || isa<func::ReturnOp>(op)) {
      return;
    }

    if (isa<neura::ConstantOp>(op) ||
        isa<neura::ReserveOp>(op) ||
        isa<neura::GrantOnceOp>(op) ||
        isa<neura::GrantAlwaysOp>(op) ||
        isa<arith::ConstantOp>(op)) {
      return;
    }

    for (Value operand : op->getOperands()) {
      valueUsers[operand].insert(op);
    }
  });

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Dependency Graph:\n";
    for (auto& entry : valueUsers) {
      llvm::outs() << "[neura-interpreter]  Value: ";
      entry.first.print(llvm::outs());
      llvm::outs() << " -> Users: ";
      for (auto* userOp : entry.second) {
        llvm::outs() << userOp->getName() << ", ";
      }
      llvm::outs() << "\n";
    }
  }
}


void addToWorklist(Operation* op) {
  if (op == nullptr)
    return;
  if (inWorklist.lookup(op))
    return;

  worklist.push_back(op);
  inWorklist[op] = true;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter] Worklist Added: " << op->getName() << "\n";
  }
}

class Memory {
public:
    Memory(size_t size) : mem(size, 0) {
        freeList[0] = size;
    }

    template <typename T>
    T load(size_t addr) const {
        assert(validAddr(addr, sizeof(T)) && "Memory load out of bounds");
        T result;
        std::memcpy(&result, &mem[addr], sizeof(T));
        return result;
    }

    template <typename T>
    void store(size_t addr, const T& value) {
        assert(validAddr(addr, sizeof(T)) && "Memory store out of bounds");
        std::memcpy(&mem[addr], &value, sizeof(T));
    }

    size_t malloc(size_t sizeBytes) {
        for (auto it = freeList.begin(); it != freeList.end(); ++it) {
            if (it->second >= sizeBytes) {
                size_t addr = it->first;
                size_t remain = it->second - sizeBytes;
                freeList.erase(it);
                if (remain > 0) {
                    freeList[addr + sizeBytes] = remain;
                }
                allocTable[addr] = sizeBytes;
                return addr;
            }
        }
        throw std::runtime_error("Out of memory");
    }

    void free(size_t addr) {
        auto it = allocTable.find(addr);
        if (it == allocTable.end()) {
            std::cerr << "Invalid free at addr " << addr << "\n";
            return;
        }
        size_t size = it->second;
        allocTable.erase(it);
        freeList[addr] = size;

        mergeFreeBlocks();
    }

    void dump(size_t start = 0, size_t length = 64) const {
        for (size_t i = start; i < start + length && i < mem.size(); ++i) {
            printf("%02X ", mem[i]);
            if ((i - start + 1) % 16 == 0) printf("\n");
        }
        printf("\n");
    }

    size_t getSize() const { return mem.size(); }

private:
    std::vector<uint8_t> mem;
    std::unordered_map<size_t, size_t> allocTable;
    std::map<size_t, size_t> freeList;  

    bool validAddr(size_t addr, size_t size) const {
        return addr + size <= mem.size();
    }

    void mergeFreeBlocks() {
        auto it = freeList.begin();
        while (it != freeList.end()) {
            auto curr = it++;
            if (it != freeList.end() && curr->first + curr->second == it->first) {
                curr->second += it->second;
                freeList.erase(it);
                it = curr;
            }
        }
    }
};

template <typename T>
class NdArray {
public:
    NdArray(Memory& mem, const std::vector<size_t>& dims)
        : memory(mem), shape(dims) {
        strides.resize(dims.size());
        size_t stride = sizeof(T);
        for (int i = dims.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= dims[i];
        }
        totalSize = stride;
        baseAddr = memory.malloc(totalSize);
    }

    void set(const std::vector<size_t>& indices, T value) {
        size_t addr = calcAddr(indices);
        memory.store<T>(addr, value);
    }

    T get(const std::vector<size_t>& indices) const {
        size_t addr = calcAddr(indices);
        return memory.load<T>(addr);
    }

    void dumpRaw() const {
        memory.dump(baseAddr, totalSize);
    }

    void free() {
        memory.free(baseAddr);
    }

private:
    Memory& memory;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t totalSize;
    size_t baseAddr;

    size_t calcAddr(const std::vector<size_t>& indices) const {
        assert(indices.size() == shape.size());
        size_t offset = baseAddr;
        for (size_t i = 0; i < indices.size(); ++i) {
            assert(indices[i] < shape[i]);
            offset += indices[i] * strides[i];
        }
        return offset;
    }
};

// Data structure to hold both value and predicate.
struct PredicatedData {
  float value;
  bool predicate;
  bool isVector;
  std::vector<float> vectorData;
  bool isReserve;
  bool isUpdated;
};

bool handleArithConstantOp(mlir::arith::ConstantOp op, llvm::DenseMap<Value, PredicatedData>& valueMap) {
  auto attr = op.getValue();
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing arith.constant:\n";
  }

  PredicatedData val{0.0f, true};  // arith constants always have true predicate
      
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    val.value = floatAttr.getValueAsDouble();
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Parsed float constant : " 
                   << llvm::format("%.6f", val.value) << "\n";
    }
  } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    if(intAttr.getType().isInteger(1)) {
      val.value = intAttr.getInt() != 0 ? 1.0f : 0.0f; 
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  └─ Parsed boolean constant : " 
                     << (val.value ? "true" : "false") << "\n";
      }
    } else {
      val.value = static_cast<float>(intAttr.getInt());
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  └─ Parsed integer constant : " 
                     << llvm::format("%.6f", val.value) << "\n";
      }
    }
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported constant type in arith.constant\n";
    }
    return false;
  }

  assert(valueMap.count(op.getResult()) == 0 && "Duplicate constant result?");
  valueMap[op.getResult()] = val;
  return true;
}

bool handleNeuraConstantOp(neura::ConstantOp op, llvm::DenseMap<Value, PredicatedData>& valueMap) {
  auto attr = op.getValue();

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.constant:\n";
  }
  
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    PredicatedData val;
    val.value = floatAttr.getValueAsDouble();
    val.predicate = true;
    val.isVector = false;
            
    if (auto predAttr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = predAttr.getValue();
    }
       
    assert(valueMap.count(op.getResult()) == 0 && "Duplicate constant result?");
    valueMap[op.getResult()] = val;
  } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    PredicatedData val;
    val.value = static_cast<float>(intAttr.getInt());
    val.predicate = true;
    val.isVector = false;
      
    if (auto predAttr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = predAttr.getValue();
    }
    
    assert(valueMap.count(op.getResult()) == 0 && "Duplicate constant result?");       
    valueMap[op.getResult()] = val;
  } else if (auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!denseAttr.getElementType().isF32()) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported vector element type in neura.constant\n";
      }
      return false;
    }
            
    PredicatedData val;
    val.isVector = true;
    val.predicate = true;
          
    size_t vectorSize = denseAttr.getNumElements();
    val.vectorData.resize(vectorSize);
            
    auto floatValues = denseAttr.getValues<float>();
    std::copy(floatValues.begin(), floatValues.end(), val.vectorData.begin());
            
    if (auto predAttr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = predAttr.getValue();
    }
     
    assert(valueMap.count(op.getResult()) == 0 && "Duplicate constant result?");
    valueMap[op.getResult()] = val;
    
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Parsed vector constant of size: " << vectorSize << "\n";
    }
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported constant type in neura.constant\n";
    }
    return false;
  }
  return true;
}

bool handleAddOp(neura::AddOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.add:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.add expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  int64_t lhsInt = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhsInt = static_cast<int64_t>(std::round(rhs.value));
  bool finalPredicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  int64_t sum = lhsInt + rhsInt;

  PredicatedData result;
  result.value = static_cast<float>(sum);
  result.predicate = finalPredicate;
  result.isVector = false;

  valueMap[op.getResult()] = result;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n";
  }

  return true;
}

bool handleSubOp(neura::SubOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.sub:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.sub expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getOperand(0)];
  auto rhs = valueMap[op.getOperand(1)];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  bool finalPredicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  int64_t lhsInt = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhsInt = static_cast<int64_t>(std::round(rhs.value));
  int64_t resultInt = lhsInt - rhsInt;

  PredicatedData result;
  result.value = static_cast<float>(resultInt);
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n";
  } 

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFAddOp(neura::FAddOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fadd:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fadd expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];
  bool finalPredicate = lhs.predicate && rhs.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  PredicatedData result;
  result.value = lhs.value + rhs.value;
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n"; 
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFSubOp(neura::FSubOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fsub:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fsub expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];
  bool finalPredicate = lhs.predicate && rhs.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }
  
  PredicatedData result;
  result.value = lhs.value - rhs.value;
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n"; 
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFMulOp(neura::FMulOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fmul:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fmul expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getOperand(0)];
  auto rhs = valueMap[op.getOperand(1)];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  bool finalPredicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  float lhsFloat = static_cast<float>(lhs.value);
  float rhsFloat = static_cast<float>(rhs.value);
  float resultFloat = lhsFloat * rhsFloat;

  PredicatedData result;
  result.value = resultFloat;
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFDivOp(neura::FDivOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fdiv:\n";
  }
  
  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fdiv expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getOperand(0)];
  auto rhs = valueMap[op.getOperand(1)];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  bool finalPredicate = lhs.predicate && rhs.predicate;
  
  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  float resultFloat = 0.0f;
  float rhsFloat = static_cast<float>(rhs.value);

  if (rhsFloat == 0.0f) {
    resultFloat = std::numeric_limits<float>::quiet_NaN();
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Warning: Division by zero, result is NaN\n";
    }
  } else {
    float lhsFloat = static_cast<float>(lhs.value);
    resultFloat = lhsFloat / rhsFloat;
  }

  PredicatedData result;
  result.value = resultFloat;
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleVFMulOp(neura::VFMulOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.vfmul:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.vfmul expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  if (!lhs.isVector || !rhs.isVector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.vfmul requires both operands to be vectors\n";
    }
    return false;
  }

  auto printVector = [](ArrayRef<float> vec) {
    llvm::outs() << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      llvm::outs() << vec[i];
      if (i != vec.size() - 1)
        llvm::outs() << ", ";
    }
    llvm::outs() << "]";
  };

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : vector size = " << lhs.vectorData.size() << ", ";
    printVector(lhs.vectorData);
    llvm::outs() << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : vector size = " << rhs.vectorData.size() << ", ";
    printVector(rhs.vectorData);
    llvm::outs() << ", [pred = " << rhs.predicate << "]\n";
  }

  if (lhs.vectorData.size() != rhs.vectorData.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Vector size mismatch in neura.vfmul\n";
    }
    return false;
  }

  bool finalPredicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    if (pred.isVector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Predicate operand must be a scalar in neura.vfmul\n";
      }
      return false;
    }
    finalPredicate = finalPredicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  PredicatedData result;
  result.isVector = true;
  result.predicate = finalPredicate;
  result.vectorData.resize(lhs.vectorData.size());

  for (size_t i = 0; i < lhs.vectorData.size(); ++i) {
    result.vectorData[i] = lhs.vectorData[i] * rhs.vectorData[i];
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : " << "vector size = " << result.vectorData.size() << ", ";
    printVector(result.vectorData); 
    llvm::outs() << ", [pred = " << result.predicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFAddFAddOp(neura::FAddFAddOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fadd_fadd:\n";
  }

  if (op.getNumOperands() < 3) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fadd_fadd expects at least three operands\n";
    }
    return false;
  }

  auto a = valueMap[op.getA()];
  auto b = valueMap[op.getB()];
  auto c = valueMap[op.getC()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand A : value = " << a.value << ", [pred = " << a.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand B : value = " << b.value << ", [pred = " << b.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Operand C : value = " << c.value << ", [pred = " << c.predicate << "]\n";
  }

  bool finalPredicate = a.predicate && b.predicate && c.predicate;

  if (op.getNumOperands() > 3) {
    auto predOperand = valueMap[op.getOperand(3)];
    finalPredicate = finalPredicate && predOperand.predicate && (predOperand.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << predOperand.value 
                   << " [pred = " << predOperand.predicate << "]\n";
    }
  }

  float resultValue = (a.value + b.value) + c.value;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Calculation  : (" << a.value << " + " << b.value << ") + " << c.value 
                 << " = " << resultValue << "\n";
  } 

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result       : value = " << resultValue 
                 << ", [pred = " << finalPredicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFMulFAddOp(neura::FMulFAddOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fmul_fadd:\n";
  }
  if (op.getNumOperands() < 3) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fmul_fadd expects at least three operands\n";
    }
    return false;
  }

  auto a = valueMap[op.getA()];
  auto b = valueMap[op.getB()];
  auto c = valueMap[op.getC()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand A : value = " << a.value << ", [pred = " << a.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand B : value = " << b.value << ", [pred = " << b.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Operand C : value = " << c.value << ", [pred = " << c.predicate << "]\n";
  }

  bool finalPredicate = a.predicate && b.predicate && c.predicate;

  if (op.getNumOperands() > 3) {
    auto predOperand = valueMap[op.getOperand(3)];
    finalPredicate = finalPredicate && predOperand.predicate && (predOperand.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << predOperand.value 
                   << ", [pred = " << predOperand.predicate << "]\n";
    }
  }

  float resultValue = 0.0f;
  float mulResult = a.value * b.value;
  resultValue = mulResult + c.value;
  
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Calculation  : (" << a.value << " * " << b.value << ") + " << c.value 
                 << " = " << mulResult << " + " << c.value << " = " << resultValue << "\n";
  }

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result       : value = " << resultValue 
                 << ", [pred = " << finalPredicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFuncReturnOp(func::ReturnOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing func.return:\n";
  }
  if (!op && isVerboseMode()) {
    llvm::errs() << "[neura-interpreter]  └─ Expected func.return but got something else\n";
    return false;
  }

  if (op.getNumOperands() == 0) {
    llvm::outs() << "[neura-interpreter]  → Output: (void)\n";
    return true;
  }

  auto result = valueMap[op.getOperand(0)];
  if (result.isVector) {
          
    llvm::outs() << "[neura-interpreter]  → Output: ["; 
    for (size_t i = 0; i < result.vectorData.size(); ++i) {
      float val = result.predicate ? result.vectorData[i] : 0.0f;
      llvm::outs() << llvm::format("%.6f", val);
      if (i != result.vectorData.size() - 1)
        llvm::outs() << ", ";
      }
      llvm::outs() << "]\n";
  } else {
    float val = result.predicate ? result.value : 0.0f;
    llvm::outs() << "[neura-interpreter]  → Output: " << llvm::format("%.6f", val) << "\n";
  }
  return true;
}

bool handleFCmpOp(neura::FCmpOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fcmp:\n";
  }
  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {  
      llvm::errs() << "[neura-interpreter]  └─ neura.fcmp expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS               : value = " 
               << lhs.value << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS               : value = " 
               << rhs.value << ", [pred = " << rhs.predicate << "]\n";
  }

  bool pred = true;
  if (op.getNumOperands() > 2) {
    auto predData = valueMap[op.getPredicate()];
    pred = predData.predicate && (predData.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = " << predData.value 
                 << ", [pred = " << predData.predicate << "]\n";
    }
  }

  bool fcmpResult = false;
  StringRef cmpType = op.getCmpType();

  if (cmpType == "eq") {
    fcmpResult = (lhs.value == rhs.value);
  } else if (cmpType == "ne") {
    fcmpResult = (lhs.value != rhs.value);
  } else if (cmpType == "le") {
    fcmpResult = (lhs.value <= rhs.value);
  } else if (cmpType == "lt") {
    fcmpResult = (lhs.value < rhs.value);
  } else if (cmpType == "ge") {
    fcmpResult = (lhs.value >= rhs.value);
  } else if (cmpType == "gt") {
    fcmpResult = (lhs.value > rhs.value);
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported comparison type: " << cmpType << "\n";
    }
    return false;
  }

  bool finalPredicate = lhs.predicate && rhs.predicate && pred;
  float resultValue = fcmpResult ? 1.0f : 0.0f;

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Comparison type   : " << op.getCmpType() << "\n";  
    llvm::outs() << "[neura-interpreter]  │  └─ Comparison result : " 
                 << (fcmpResult ? "true" : "false") << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result               : value = " 
                 << resultValue << ", [pred = " << finalPredicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleICmpOp(neura::ICmpOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.icmp:\n";
  }
  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.icmp expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS               : value = " << lhs.value 
                 << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS               : value = " << rhs.value 
                 << ", [pred = " << rhs.predicate << "]\n";
  }

  bool pred = true;
  if (op.getNumOperands() > 2) {
    auto predData = valueMap[op.getPredicate()];
    pred = predData.predicate && (predData.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = " << predData.value 
                 << ", [pred = " << predData.predicate << "]\n";
    }
  }

  int64_t s_lhs = static_cast<int64_t>(std::round(lhs.value));
  int64_t s_rhs = static_cast<int64_t>(std::round(rhs.value));

  auto signed_to_unsigned = [](int64_t val) {
    return val >= 0 ? 
           static_cast<uint64_t>(val) : 
           static_cast<uint64_t>(UINT64_MAX + val + 1);
  };

  uint64_t u_lhs = signed_to_unsigned(s_lhs);
  uint64_t u_rhs = signed_to_unsigned(s_rhs);

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Signed values     : LHS = " << s_lhs 
                 << ", RHS = " << s_rhs << "\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Unsigned values   : LHS = " << u_lhs 
                 << ", RHS = " << u_rhs << "\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Comparison type   : " << op.getCmpType() << "\n";
  }

  bool icmp_result = false;
  StringRef cmp_type = op.getCmpType();

  if (cmp_type == "eq") {
    icmp_result = (s_lhs == s_rhs);
  } else if (cmp_type == "ne") {
    icmp_result = (s_lhs != s_rhs);
  } else if (cmp_type.starts_with("s")) {
    if (cmp_type == "slt") icmp_result = (s_lhs < s_rhs);
    else if (cmp_type == "sle") icmp_result = (s_lhs <= s_rhs);
    else if (cmp_type == "sgt") icmp_result = (s_lhs > s_rhs);
    else if (cmp_type == "sge") icmp_result = (s_lhs >= s_rhs);
    else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported signed comparison type: " << cmp_type << "\n";
      }
        return false;
    }
  } else if (cmp_type.starts_with("u")) {
    if (cmp_type == "ult") icmp_result = (u_lhs < u_rhs);
    else if (cmp_type == "ule") icmp_result = (u_lhs <= u_rhs);
    else if (cmp_type == "ugt") icmp_result = (u_lhs > u_rhs);
    else if (cmp_type == "uge") icmp_result = (u_lhs >= u_rhs);
    else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported unsigned comparison type: " << cmp_type << "\n";
      }
      return false;
    }
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported comparison type: " << cmp_type << "\n";
    }
    return false;
  }

  bool finalPredicate = lhs.predicate && rhs.predicate && pred;
  float resultValue = icmp_result ? 1.0f : 0.0f;

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  │  └─ Comparison result : " << (icmp_result ? "true" : "false") << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result               : value = " << resultValue 
                 << ", [pred = " << finalPredicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleOrOp(neura::OrOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.or:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.or expects at least two operands\n";
    }
    return false;
  }

  auto lhs = valueMap[op.getOperand(0)];
  auto rhs = valueMap[op.getOperand(1)];

  if (lhs.isVector || rhs.isVector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.or requires scalar operands\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS        : value = " << lhs.value << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS        : value = " << rhs.value << ", [pred = " << rhs.predicate << "]\n";
  }

  int64_t lhsInt = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhsInt = static_cast<int64_t>(std::round(rhs.value));
  int64_t resultInt = lhsInt | rhsInt;

  bool finalPredicate = lhs.predicate && rhs.predicate;
  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = " << pred.value 
                 << ", [pred = " << pred.predicate << "]\n";
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Bitwise OR : " << lhsInt;
    if (lhsInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
    llvm::outs() << " | " << rhsInt;
    if (rhsInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
    llvm::outs() << " = " << resultInt;
    if (resultInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
    llvm::outs() << "\n";
  }

  PredicatedData result;
  result.value = static_cast<float>(resultInt);
  result.predicate = finalPredicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result     : value = " << result.value 
                 << ", [pred = " << finalPredicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleNotOp(neura::NotOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {

  auto input = valueMap[op.getOperand()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.not:\n";
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Input       : value = " << input.value 
                 << ", [pred = " << input.predicate << "]\n";
  }

  int64_t inputInt = static_cast<int64_t>(std::round(input.value));
  int64_t resultInt = !inputInt;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Logical NOT : !" << inputInt;
    if (inputInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
    llvm::outs() << " = " << resultInt;
    if (resultInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
    llvm::outs() << "\n";
  }


  PredicatedData result;
  result.value = static_cast<float>(resultInt);
  result.predicate = input.predicate;
  result.isVector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result         : value = " << result.value 
                 << ", [pred = " << result.predicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleSelOp(neura::SelOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.sel:\n";
  }

  if (op.getNumOperands() != 3) {  
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.sel expects exactly 3 operands (cond, ifTrue, ifFalse)\n";
    }
    return false;
  }

  auto cond = valueMap[op.getCond()];
  auto ifTrue = valueMap[op.getIfTrue()];
  auto ifFalse = valueMap[op.getIfFalse()];
  bool condValue = (cond.value != 0.0f) && cond.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Condition : value = " << cond.value 
                 << ", [pred = " << cond.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ If true   : value = " << ifTrue.value 
                 << ", [pred = " << ifTrue.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ If false  : value = " << ifFalse.value 
                 << ", [pred = " << ifFalse.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation \n"; 
  }

  PredicatedData result;
  if (condValue) {
    result.value = ifTrue.value;
    result.predicate = ifTrue.predicate && cond.predicate;  
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Condition is true, selecting 'ifTrue' branch\n";
    }
  } else {
    result.value = ifFalse.value;
    result.predicate = ifFalse.predicate && cond.predicate; 
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Condition is false, selecting 'ifFalse' branch\n";
    }
  }

  result.isVector = ifTrue.isVector && ifFalse.isVector; 

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result      : value = " << result.value 
                 << ", predicate = " << result.predicate << "\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleCastOp(neura::CastOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.cast:\n";
  }
  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cast expects 1 or 2 operands\n";
    }
    return false;
  }

  auto input = valueMap[op.getOperand(0)];
  std::string castType = op.getCastType().str();

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Input  : value = " 
                 << input.value << ", [pred = " << input.predicate << "]\n";
  }

  bool finalPredicate = input.predicate;
  if (op.getOperation()->getNumOperands() > 1) {
    auto predOperand = valueMap[op.getOperand(1)];
    finalPredicate = finalPredicate && predOperand.predicate && (predOperand.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n"; 
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << predOperand.value 
                   << ", [pred = " << predOperand.predicate << "]\n";
    }
  }
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Cast type : " << castType << "\n";
  } 

  float resultValue = 0.0f;
  auto inputType = op.getOperand(0).getType();

    if (castType == "f2i") {
      if (!inputType.isF32()) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Cast type 'f2i' requires f32 input\n";
        }
        return false;
      }
      int64_t intValue = static_cast<int64_t>(std::round(input.value));
      resultValue = static_cast<float>(intValue);
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ Converting float to integer " 
        << input.value << " -> " << intValue << "\n";
      }

    } else if (castType == "i2f") {
      if (!inputType.isInteger()) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Cast type 'i2f' requires integer input\n";
        }
        return false;
      }
      int64_t intValue = static_cast<int64_t>(input.value);
      resultValue = static_cast<float>(intValue);
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ Converting integer to float " 
                     << intValue << " -> " << resultValue << "\n";
      }
    } else if (castType == "bool2i" || castType == "bool2f") {
      if (!inputType.isInteger(1)) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Cast type '" << castType 
                       << "' requires i1 (boolean) input\n";
        }
        return false;
      }
      bool boolValue = (input.value != 0.0f);
      resultValue = boolValue ? 1.0f : 0.0f;
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ Converting boolean to number " 
                     << (boolValue ? "true" : "false") << " -> " << resultValue << "\n";
      }
    } else if (castType == "i2bool" || castType == "f2bool") {
      if (!inputType.isInteger() && !inputType.isF32()) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Cast type '" << castType 
                       << "' requires integer or f32 input\n";
        }
        return false;
      }
      bool boolValue = (input.value != 0.0f);
      resultValue = boolValue ? 1.0f : 0.0f;
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ Converting number to boolean " 
                    << input.value << " -> " << (boolValue ? "true" : "false") << " (stored as " << resultValue << ")\n";
      }
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported cast type: " << castType << "\n";
      }
      return false;
    }

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = input.isVector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result    : value = " << resultValue 
                 << ", [pred = " << finalPredicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleLoadOp(neura::LoadOp op, llvm::DenseMap<Value, PredicatedData> &valueMap, Memory &mem) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.load:\n";
  }

  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.load expects 1 or 2 operands (address, [predicate])\n";
    }
    return false;
  }

  auto addrVal = valueMap[op.getOperand(0)];
  bool finalPredicate = addrVal.predicate;

  if (op.getNumOperands() > 1) {
    auto predVal = valueMap[op.getOperand(1)];
    finalPredicate = finalPredicate && predVal.predicate && (predVal.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n"; 
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << predVal.value 
                   << ", [pred = " << predVal.predicate << "]\n";
    }
  }

  float val = 0.0f;
  size_t addr = static_cast<size_t>(addrVal.value);

  if (finalPredicate) {
    auto resultType = op.getResult().getType();
    if (resultType.isF32()) {
      val = mem.load<float>(addr);
    } else if (resultType.isInteger(32)) {
      val = static_cast<float>(mem.load<int32_t>(addr));
    } else if (resultType.isInteger(1)) {
      val = static_cast<float>(mem.load<bool>(addr));
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported load type\n";
      }
      return false;
    }
  } else {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Load skipped due to [pred = 0]\n";
    }
    val = 0.0f; // Default value when load is skipped
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Load [addr = " << addr << "] => val = "
                 << val << ", [pred = " << finalPredicate << "]\n";
  }

  valueMap[op.getResult()] = { val, finalPredicate };
  return true;
}

bool handleStoreOp(neura::StoreOp op, llvm::DenseMap<Value, PredicatedData> &valueMap, Memory &mem) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.store:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.store expects at least two operands (value, address)\n";
    }
    return false;
  }

  auto valData = valueMap[op.getOperand(0)];
  auto addrVal = valueMap[op.getOperand(1)];
  bool finalPredicate = addrVal.predicate;

  if (op.getNumOperands() > 2) {
    auto predVal = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && predVal.predicate && (predVal.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n"; 
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << predVal.value 
                   << ", [pred = " << predVal.predicate << "]\n";
    }
  }

  size_t addr = static_cast<size_t>(addrVal.value);

  if(finalPredicate) {
    auto valType = op.getOperand(0).getType();
    if (valType.isF32()) {
      mem.store<float>(addr, valData.value);
    } else if (valType.isInteger(32)) {
      mem.store<int32_t>(addr, static_cast<int32_t>(valData.value));
    } else if (valType.isInteger(1)) {
    mem.store<bool>(addr, (valData.value != 0.0f));
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported store type\n";
      }
      return false;
    }
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Store [addr = " << addr 
                   << "] => val = " << valData.value 
                   << ", [pred = 1" << "]\n";
    }
  } else {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Store skipped due to [pred = 0]\n";
    }
  }

  return true;
}

bool handleGEPOp(neura::GEP op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.gep:\n";
  }

  if (op.getOperation()->getNumOperands() < 1) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.gep expects at least 1 operand (base address)\n";
    }
    return false;
  }

  auto baseVal = valueMap[op.getOperand(0)];
  size_t baseAddr = static_cast<size_t>(baseVal.value);
  bool finalPredicate = baseVal.predicate;

  if(isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Base address: value = " << baseAddr << ", [pred = " << baseVal.predicate << "]\n";
  }

  unsigned numOperands = op.getOperation()->getNumOperands();
  bool hasPredicate = false;
  unsigned indexCount = numOperands - 1;  

  if (numOperands > 1) {
    auto lastOperandType = op.getOperand(numOperands - 1).getType();
    if (lastOperandType.isInteger(1)) {
      hasPredicate = true;
      indexCount -= 1; 
    }
  }

  auto stridesAttr = op->getAttrOfType<mlir::ArrayAttr>("strides");
  if (!stridesAttr) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.gep requires 'strides' attribute\n";
    }
    return false;
  }

  std::vector<size_t> strides;
  for (auto s : stridesAttr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(s);
    if (!intAttr) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Invalid type in 'strides' attribute (expected integer)\n";
      }  
      return false;
    }
    strides.push_back(static_cast<size_t>(intAttr.getInt()));
  }

  if (indexCount != strides.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ GEP index count (" << indexCount 
                   << ") mismatch with strides size (" << strides.size() << ")\n";
    }
    return false;
  }

  size_t offset = 0;
  for (unsigned i = 0; i < indexCount; ++i) {
    auto idxVal = valueMap[op.getOperand(i + 1)]; 
    if (!idxVal.predicate) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ GEP index " << i << " has false predicate\n";
      }
      return false;
    }

    size_t idx = static_cast<size_t>(idxVal.value);
    offset += idx * strides[i];
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Index " << i << ": value = " << idx << ", stride = " << strides[i] 
                   << ", cumulative offset = " << offset << "\n";
    }
  }

  if (hasPredicate) {
    auto predVal = valueMap[op.getOperand(numOperands - 1)];
    finalPredicate = finalPredicate && predVal.predicate && (predVal.value != 0.0f);
    if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Predicate operand: value = " << predVal.value 
                 << ", [pred = " << predVal.predicate << "]\n";
    }
  }

  size_t finalAddr = baseAddr + offset;

  PredicatedData result;
  result.value = static_cast<float>(finalAddr);
  result.predicate = finalPredicate;
  result.isVector = false; 

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Final GEP result: base = " << baseAddr << ", total offset = " << offset 
                 << ", final address = " << finalAddr 
                 << ", [pred = " << finalPredicate << "]\n";    
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleLoadIndexedOp(neura::LoadIndexedOp op,
                         llvm::DenseMap<Value, PredicatedData> &valueMap,
                         Memory &mem) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Executing neura.load_indexed:\n";
  }

  auto baseVal = valueMap[op.getBase()];
  if (baseVal.isVector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Vector base not supported in load_indexed\n";
    }
    return false;
  }
  float baseF = baseVal.value;
  bool finalPredicate = baseVal.predicate;

  float offset = 0.0f;
  for (Value idx : op.getIndices()) {
    auto idxVal = valueMap[idx];
    if (idxVal.isVector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Vector index not supported in load_indexed\n";
      }
      return false;
    }
    offset += idxVal.value;
    finalPredicate = finalPredicate && idxVal.predicate;
  }

  if (op.getPredicate()) {
    Value predOperand = op.getPredicate();
    auto predVal = valueMap[predOperand];
    if (predVal.isVector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Vector predicate not supported\n";
      }
      return false;
    }
    finalPredicate = finalPredicate && predVal.predicate && (predVal.value != 0.0f);
  }

  size_t addr = static_cast<size_t>(baseF + offset);
  float val = 0.0f;

  if (finalPredicate) {
    auto resultType = op.getResult().getType();
    if (resultType.isF32()) {
      val = mem.load<float>(addr);
    } else if (resultType.isInteger(32)) {
      val = static_cast<float>(mem.load<int32_t>(addr));
    } else if (resultType.isInteger(1)) {
      val = static_cast<float>(mem.load<bool>(addr));
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported result type\n";
      }
      return false;
    }
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ LoadIndexed [addr = " << addr << "] => val = "
                 << val << ", [pred = " << finalPredicate << "]\n";
  }

  valueMap[op.getResult()] = { val, finalPredicate, false, {}, false };
  return true;
}

bool handleStoreIndexedOp(neura::StoreIndexedOp op,
                          llvm::DenseMap<Value, PredicatedData> &valueMap,
                          Memory &mem) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.store_indexed:\n";
  }

  auto valToStore = valueMap[op.getValue()];
  if (valToStore.isVector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Vector value not supported in store_indexed\n";
    }
    return false;
  }
  float value = valToStore.value;
  bool finalPredicate = valToStore.predicate;

  auto baseVal = valueMap[op.getBase()];
  if (baseVal.isVector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Vector base not supported in store_indexed\n";
    }
    return false;
  }
  float baseF = baseVal.value;
  finalPredicate = finalPredicate && baseVal.predicate;

  float offset = 0.0f;
  for (Value idx : op.getIndices()) {
    auto idxVal = valueMap[idx];
    if (idxVal.isVector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Vector index not supported in store_indexed\n";
      }
      return false;
    }
    offset += idxVal.value;
    finalPredicate = finalPredicate && idxVal.predicate;
  }

  if (op.getPredicate()) {
      Value predOperand = op.getPredicate();
      auto predVal = valueMap[predOperand];
      if (predVal.isVector) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Vector predicate not supported\n";
        }
        return false;
      }
      finalPredicate = finalPredicate && predVal.predicate && (predVal.value != 0.0f);
  }

  size_t addr = static_cast<size_t>(baseF + offset);

  if (finalPredicate) {
    auto valType = op.getValue().getType();
    if (valType.isF32()) {
      mem.store<float>(addr, value);
    } else if (valType.isInteger(32)) {
      mem.store<int32_t>(addr, static_cast<int32_t>(value));
    } else if (valType.isInteger(1)) {
      mem.store<bool>(addr, value != 0.0f);
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported value type in store_indexed\n";
      }
      return false;
    }
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ StoreIndexed [addr = " << addr << "] <= val = "
                 << value << ", [pred = " << finalPredicate << "]\n";
  }

  return true;
}

bool handleBrOp(neura::Br op, llvm::DenseMap<Value, PredicatedData> &valueMap, 
                Block *&currentBlock, Block *&lastVisitedBlock) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.br:\n";
  }

  Block *destBlock = op.getDest();
  if (!destBlock) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.br: Target block is null\n";
    }
    return false;
  }

  auto currentSuccsRange = currentBlock->getSuccessors();
  std::vector<Block *> succBlocks(currentSuccsRange.begin(), currentSuccsRange.end());

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Block Information\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Current block    : Block@" << currentBlock << "\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Successor blocks : \n";
    for (unsigned int i = 0; i < succBlocks.size(); ++i) {
    if(i < succBlocks.size() - 1)
      llvm::outs() << "[neura-interpreter]  │  │  ├─ [" << i << "] Block@" << succBlocks[i] << "\n";
    else
      llvm::outs() << "[neura-interpreter]  │  │  └─ [" << i << "] Block@" << succBlocks[i] << "\n";
    }
    llvm::outs() << "[neura-interpreter]  │  └─ Target block : Block@" << destBlock << "\n";
    llvm::outs() << "[neura-interpreter]  ├─ Pass Arguments\n";
  }

  const auto &args = op.getArgs();
  const auto &destParams = destBlock->getArguments();

  if (args.size() != destParams.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.br: Argument count mismatch (passed " 
                 << args.size() << ", target expects " << destParams.size() << ")\n";
    }
    return false;
  }


  for (size_t i = 0; i < args.size(); ++i) {
    Value destParam = destParams[i];
    Value srcArg = args[i];
    
    if (!valueMap.count(srcArg)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ neura.br: Argument " << i 
                     << " (source value) not found in value map\n";
      }
      return false;
    }
    
    valueMap[destParam] = valueMap[srcArg];
    if (isVerboseMode() && i < destParams.size() - 1) {
      llvm::outs() << "[neura-interpreter]  │  ├─ Param[" << i << "]: value = " 
                   << valueMap[srcArg].value << "\n";
    } else if (isVerboseMode() && i == destParams.size() - 1) {
      llvm::outs() << "[neura-interpreter]  │  └─ Param[" << i << "]: value = " 
                   << valueMap[srcArg].value << "\n";
    }
  }

  lastVisitedBlock = currentBlock;
  currentBlock = destBlock;
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Control Transfer\n";
    llvm::outs() << "[neura-interpreter]     └─ Jump successfully to Block@ " << destBlock << "\n";
  }
  return true;
}

bool handleCondBrOp(neura::CondBr op, llvm::DenseMap<Value, PredicatedData> &valueMap, 
                    Block *&currentBlock, Block *&lastVisitedBlock) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.cond_br:\n";
  }

  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br expects 1 or 2 operands (condition + optional predicate)\n";
    }
    return false;
  }

  auto condValue = op.getCondition();
  if (!valueMap.count(condValue)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ cond_br: condition value not found in valueMap! (SSA name missing)\n";
    }
    return false;
  }
  auto condData = valueMap[op.getCondition()];

  if (!op.getCondition().getType().isInteger(1)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: condition must be of type i1 (boolean)\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Condition     : value = " << condData.value 
               << ", [pred = " << condData.predicate << "]\n";
  }

  bool finalPredicate = condData.predicate;
  if (op.getNumOperands() > 1) {
    auto predData = valueMap[op.getPredicate()];
    finalPredicate = finalPredicate && predData.predicate && (predData.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << predData.value
                   << " [pred = " << predData.predicate << "]\n";
    }
  }

  auto currentSuccsRange = currentBlock->getSuccessors();
  std::vector<Block *> succBlocks(currentSuccsRange.begin(), currentSuccsRange.end());

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Block Information\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Current block : Block@" << currentBlock << "\n";
    llvm::outs() << "[neura-interpreter]  ├─ Branch Targets\n";
    for (unsigned int i = 0; i < succBlocks.size(); ++i) {
    if(i < succBlocks.size() - 1)
      llvm::outs() << "[neura-interpreter]  │  ├─ True block    : Block@" << succBlocks[i] << "\n";
    else
      llvm::outs() << "[neura-interpreter]  │  └─ False block   : Block@" << succBlocks[i] << "\n";
    }
  }

  if (!finalPredicate) {
    llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: condition or predicate is invalid\n";
    return false;
  }

  bool isTrueBranch = (condData.value != 0.0f);
  Block *targetBlock = isTrueBranch ? op.getTrueDest() : op.getFalseDest();
  const auto &branchArgs = isTrueBranch ? op.getTrueArgs() : op.getFalseArgs();
  const auto &targetParams = targetBlock->getArguments();
  
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Condition is " << (condData.value != 0.0f ? "true" : "false")
                 << " → selecting '" << (isTrueBranch ? "true" : "false") << "' branch\n";
  }
  

  if (branchArgs.size() != targetParams.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: argument count mismatch for " 
                   << (isTrueBranch ? "true" : "false") << " branch (expected " 
                   << targetParams.size() << ", got " << branchArgs.size() << ")\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    if (!branchArgs.empty()) {
      llvm::outs() << "[neura-interpreter]  ├─ Pass Arguments\n";
    }
    for (size_t i = 0; i < branchArgs.size(); ++i) {
      valueMap[targetParams[i]] = valueMap[branchArgs[i]];
      if (i < branchArgs.size() - 1)
        llvm::outs() << "[neura-interpreter]  │  ├─ param[" << i << "]: value = " 
                     << valueMap[branchArgs[i]].value << "\n";
      else {
        llvm::outs() << "[neura-interpreter]  │  └─ param[" << i << "]: value = " 
                     << valueMap[branchArgs[i]].value << "\n";
      }
    }
  }



  lastVisitedBlock = currentBlock;
  currentBlock = targetBlock;
  
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Control Transfer\n";
    llvm::outs() << "[neura-interpreter]     └─ Jump successfully to Block@" << targetBlock << "\n";
  }

  return true;
}

bool handlePhiOpControlFlowMode(neura::PhiOp op, llvm::DenseMap<Value, PredicatedData> &valueMap, 
                 Block *currentBlock, Block *lastVisitedBlock) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.phi:\n";
  }

  auto predecessorsRange = currentBlock->getPredecessors();
  std::vector<Block*> predecessors(predecessorsRange.begin(), predecessorsRange.end());
  size_t predCount = predecessors.size(); 

  if (predCount == 0) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Current block has no predecessors\n";
    }
    return false;
  }

  size_t predIndex = 0;
  bool found = false;
  for (auto pred : predecessors) {
    if (pred == lastVisitedBlock) {
      found = true;
      break;
    }
    ++predIndex;
  }

  if (!found) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Last visited block not found in predecessors\n";
    }
    return false;
  }

  auto inputs = op.getInputs();
  size_t inputCount = inputs.size();

  if (inputCount != predCount) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Input count (" << inputCount 
                   << ") != predecessor count (" << predCount << ")\n";
    }
    return false;
  }

  if (predIndex >= inputCount) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Invalid predecessor index (" << predIndex << ")\n";
    }
    return false;
  }

  Value inputVal = inputs[predIndex];
  if (!valueMap.count(inputVal)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Input value not found in value map\n";
    }
    return false;
  }

  PredicatedData inputData = valueMap[inputVal];
  valueMap[op.getResult()] = inputData;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Predecessor blocks (" << predCount << ")\n";
    for (size_t i = 0; i < predCount; ++i) {
      if(i < predCount - 1) {
        llvm::outs() << "[neura-interpreter]  │  ├─ [" << i << "]: " << "Block@" << predecessors[i];
      } else {
        llvm::outs() << "[neura-interpreter]  │  └─ [" << i << "]: " << "Block@" << predecessors[i];
      }
      if (i == predIndex) {
        llvm::outs() << " (→ current path)\n";
      } else {
        llvm::outs() << "\n";
      }
    }
    llvm::outs() << "[neura-interpreter]  └─ Result    : " << op.getResult() << "\n";
    llvm::outs() << "[neura-interpreter]     └─ Value : " << inputData.value 
                 << ", [Pred = " << inputData.predicate << "]\n";
  }
  return true;
}

bool handlePhiOpDataFlowMode(neura::PhiOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.phi(dataflow):\n";
  }

  auto inputs = op.getInputs();
  size_t inputCount = inputs.size();

  if (inputCount == 0) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: No inputs provided (execution failed)\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Input values (" << inputCount << ")\n";
    for (size_t i = 0; i < inputCount; ++i) {
      Value input = inputs[i];
      if (valueMap.count(input)) {
        auto inputData = valueMap[input];
        const std::string prefix = (i < inputCount - 1) ? "│ ├─" : "│ └─";
        llvm::outs() << "[neura-interpreter]  " << prefix << "[" << i << "]:"
                     << "value = " << inputData.value << ","
                     << "pred = " << inputData.predicate << "\n";
      } else {
        const std::string prefix = (i < inputCount - 1) ? "│ ├─" : "│ └─";
        llvm::outs() << "[neura-interpreter]  " << prefix << "[" << i << "]: <undefined>\n";
      }
    }
  }

  PredicatedData result;
  result.value = 0.0f;
  result.predicate = false;
  result.isVector = false;
  result.vectorData = {};
  result.isReserve = false;
  result.isUpdated = false;
  bool foundValidInput = false;

  for (size_t i = 0; i < inputCount; ++i) { 
    Value input = inputs[i];

    if (!valueMap.count(input)) {
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  ├─ Input [" << i << "] not found, skipping\n";
      }
      continue;
    }

    auto inputData = valueMap[input];

    if (inputData.predicate && !foundValidInput) {
      result.value = inputData.value;
      result.predicate = inputData.predicate;
      result.isVector = inputData.isVector;
      result.vectorData = inputData.vectorData;
      foundValidInput = true;

      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  ├─ Selected input [" << i << "] (latest valid)\n";
      }
      break;
    }
  }

  if (!foundValidInput && inputCount > 0) {
    Value firstInput = inputs[0];
    if (valueMap.count(firstInput)) {
      auto firstData = valueMap[firstInput];
      result.value = firstData.value;
      result.isVector = firstData.isVector;
      result.vectorData = firstData.vectorData;
      result.predicate = false;
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  ├─ No valid input, using first input with pred=false\n";
      }
    }
  }

  bool fieldsChanged = false;
  if (valueMap.count(op.getResult())) {
    auto oldResult = valueMap[op.getResult()];
    fieldsChanged = (result.value != oldResult.value) ||
                    (result.predicate != oldResult.predicate) ||
                    (result.isVector != oldResult.isVector) ||
                    (result.vectorData != oldResult.vectorData);
  } else {
    fieldsChanged = true;
  }

  result.isUpdated = fieldsChanged;
  valueMap[op.getResult()] = result;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Execution " << (foundValidInput ? "succeeded" : "partially succeeded")
                 << " | Result: value = " << result.value
                 << ", pred = " << result.predicate
                 << ", isUpdated = " << result.isUpdated << "\n";
  }

  return true;
}


bool handleReserveOp(neura::ReserveOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.reserve:\n";
  }

  PredicatedData placeholder;
  placeholder.value = 0.0f;
  placeholder.predicate = false;
  placeholder.isReserve = true;

  Value result = op.getResult();
  valueMap[result] = placeholder;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Created placeholder  : " << result << "\n";
    llvm::outs() << "[neura-interpreter]     ├─ Initial value     : 0.0f\n";
    llvm::outs() << "[neura-interpreter]     ├─ Initial predicate : false\n";
    llvm::outs() << "[neura-interpreter]     └─ Type              : " << result.getType() << "\n";
  }


  return true;
}

bool handleCtrlMovOp(neura::CtrlMovOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.ctrl_mov:\n";
  }

  Value source = op.getValue();
  Value target = op.getTarget();

  if (!valueMap.count(source)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.ctrl_mov: Source value not found in value map\n";
    }
    return false;
  }

  if (!valueMap.count(target) || !valueMap[target].isReserve) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.ctrl_mov: Target is not a reserve placeholder\n";
    }
    return false;
  }

  const auto &sourceData = valueMap[source];
  auto &targetData = valueMap[target];

  if (source.getType() != target.getType()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.ctrl_mov: Type mismatch (source ="
                   << source.getType() << ", target =" << target.getType() << ")\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Source: " << source <<"\n";
    llvm::outs() << "[neura-interpreter]  │  └─ value = "  << sourceData.value 
                 << ", [pred = " << sourceData.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  ├─ Target: " << target << "\n";
    llvm::outs() << "[neura-interpreter]  │  └─ value = "  << targetData.value 
                 << ", [pred = " << targetData.predicate << "]\n";
  }

  targetData.value = sourceData.value;
  targetData.predicate = sourceData.predicate;
  targetData.isVector = sourceData.isVector;
  if (sourceData.isVector) {
    targetData.vectorData = sourceData.vectorData;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Updated target placeholder\n";
    llvm::outs() << "[neura-interpreter]     └─ value = "  << targetData.value 
                 << ", [pred = " << targetData.predicate << "]\n";
  }

  return true;
}

bool handleCtrlMovOpDataFlowMode_(neura::CtrlMovOp op, 
                                 llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.ctrl_mov(dataflow):\n";
  }

  Value source = op.getValue();
  Value target = op.getTarget();

  if (!valueMap.count(source)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Source value not found in value map (execution failed)\n";
    }
    return false;
  }

  if (!valueMap.count(target) || !valueMap[target].isReserve) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Target is not a reserve placeholder (execution failed)\n";
    }
    return false;
  }

  const auto &sourceData = valueMap[source];
  auto &targetData = valueMap[target];

  float oldValue = targetData.value;
  bool oldPredicate = targetData.predicate;
  bool oldIsVector = targetData.isVector;

  targetData.value = sourceData.value;
  targetData.predicate = sourceData.predicate;
  targetData.isVector = sourceData.isVector;

  bool isTerminated = false;
  if (!isTerminated) {
    targetData.value = sourceData.value;
    targetData.predicate = sourceData.predicate;
    targetData.isVector = sourceData.isVector;
    targetData.vectorData = sourceData.isVector ? sourceData.vectorData : std::vector<float>();
  }
  
  if (sourceData.isVector) {
    targetData.vectorData = sourceData.vectorData;
  } else {
    targetData.vectorData.clear();
  }

  std::vector<float> oldVectorData = targetData.vectorData;

  bool isScalarUpdated = false;
  bool isVectorUpdated = false;
  bool isTypeUpdated = false;

  if (!isTerminated) {
    if (!targetData.isVector) {
      isScalarUpdated = (targetData.value != oldValue) || 
                       (targetData.predicate != oldPredicate) || 
                       !oldVectorData.empty();
    }
    if (targetData.isVector) {
      isVectorUpdated = (targetData.predicate != oldPredicate) || 
                       (targetData.vectorData != oldVectorData);
    }
    isTypeUpdated = (targetData.isVector != oldIsVector);
    targetData.isUpdated = isScalarUpdated || isVectorUpdated || isTypeUpdated;
  } else {
    targetData.isUpdated = false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Source: " << sourceData.value << " | " << sourceData.predicate << "\n";
    llvm::outs() << "[neura-interpreter]  ├─ Target (after update): " << targetData.value << " | " << targetData.predicate << " | isUpdated=" << targetData.isUpdated << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Execution succeeded (data copied)\n";
  }

  return true;
}

bool handleCtrlMovOpDataFlowMode(neura::CtrlMovOp op, 
                                 llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.ctrl_mov(dataflow):\n";
  }

  Value source = op.getValue();
  Value target = op.getTarget();

  if (!valueMap.count(source)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Source value not found in value map (execution failed)\n";
    }
    return false;
  }

  if (!valueMap.count(target) || !valueMap[target].isReserve) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Target is not a reserve placeholder (execution failed)\n";
    }
    return false; 
  }

  const auto &sourceData = valueMap[source];
  auto &targetData = valueMap[target];

  float oldValue = targetData.value;
  bool oldPredicate = targetData.predicate;
  bool oldIsVector = targetData.isVector;
  std::vector<float> oldVectorData = targetData.vectorData;

  bool isTerminated = false;
  bool isScalarUpdated = false;
  bool isVectorUpdated = false;
  bool isTypeUpdated = false;
  targetData.isUpdated = false;

  bool shouldUpdate = !isTerminated && (sourceData.predicate == 1);
  if (shouldUpdate) {
    targetData.value = sourceData.value;
    targetData.predicate = sourceData.predicate;
    targetData.isVector = sourceData.isVector;
    
    if (sourceData.isVector) {
      targetData.vectorData = sourceData.vectorData;
    } else {
      targetData.vectorData.clear();
    }

    if (!targetData.isVector) {
      isScalarUpdated = (targetData.value != oldValue) || 
                       (targetData.predicate != oldPredicate) || 
                       !oldVectorData.empty();
    }
    if (targetData.isVector) {
      isVectorUpdated = (targetData.predicate != oldPredicate) || 
                       (targetData.vectorData != oldVectorData);
    }
    isTypeUpdated = (targetData.isVector != oldIsVector);
    targetData.isUpdated = isScalarUpdated || isVectorUpdated || isTypeUpdated;
  } else {
    if (isVerboseMode()) {
      if (isTerminated) {
        llvm::outs() << "[neura-interpreter]  ├─ Skip update: Loop terminated\n";
      } else {
        llvm::outs() << "[neura-interpreter]  ├─ Skip update: Source predicate is invalid (pred=" << sourceData.predicate << ")\n";
      }
    }
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Source: " << sourceData.value << " | " << sourceData.predicate << "\n";
    llvm::outs() << "[neura-interpreter]  ├─ Target (after update): " << targetData.value << " | " << targetData.predicate << " | isUpdated=" << targetData.isUpdated << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Execution succeeded (update controlled by predicate and loop status)\n";
  }

  return true;
}


bool handleNeuraReturnOp(neura::ReturnOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.return:\n";
  }

  auto returnValues = op.getValues();
  if (returnValues.empty()) {
    llvm::outs() << "[neura-interpreter]  → Output: (void)\n";
    return true;
  }

  std::vector<PredicatedData> results;
  for (Value val : returnValues) {
    if (!valueMap.count(val)) {
      llvm::errs() << "[neura-interpreter]  └─ Return value not found in value map\n";
      return false;
    }
    results.push_back(valueMap[val]);
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Return values:\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto &data = results[i];
      // llvm::outs() << "[neura-interpreter]  [" << i << "]: ";
      
      if (data.isVector) {
        llvm::outs() << "[neura-interpreter]  │  └─ vector = [";
        for (size_t j = 0; j < data.vectorData.size(); ++j) {
          float val = data.predicate ? data.vectorData[j] : 0.0f;
          llvm::outs() << llvm::format("%.6f", val);
          if (j != data.vectorData.size() - 1) 
            llvm::outs() << ", ";
        }
        llvm::outs() << "]";
      } else {
        float val = data.predicate ? data.value : 0.0f;
        llvm::outs() << "[neura-interpreter]  │  └─" << llvm::format("%.6f", val);
      }
      llvm::outs() << ", [pred = " << data.predicate << "]\n";
    }
    llvm::outs() << "[neura-interpreter]  └─ Execution terminated successfully\n";
  }

  return true;
}

bool handleGrantPredicateOp(neura::GrantPredicateOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.grant_predicate:\n";
  }

  if (op.getOperation()->getNumOperands() != 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.grant_predicate expects exactly 2 operands (value, new_predicate)\n";
    } 
    return false;
  }

  if (!valueMap.count(op.getValue()) || !valueMap.count(op.getPredicate())) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Source or new predicate not found in valueMap\n";
    }
    return false;
  }

  auto source = valueMap[op.getValue()];
  auto newPred = valueMap[op.getPredicate()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Source: value = " << source.value 
                 << ", [pred = " << source.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ New predicate: value = " << newPred.value 
                 << ", [pred = " << newPred.predicate << "]\n";
  }
  
  bool isNewPredTrue = (newPred.value != 0.0f); 
  bool resultPredicate = source.predicate && newPred.predicate && isNewPredTrue;

  PredicatedData result = source;
  result.predicate = resultPredicate;
  result.isVector = source.isVector; 

  if (isVerboseMode()) {
    std::string grantStatus = resultPredicate ? "Granted access" : "Denied access (predicate false)";
    llvm::outs() << "[neura-interpreter]  ├─ " << grantStatus << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value 
                 << ", [pred = " << resultPredicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleGrantOnceOp(neura::GrantOnceOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if(isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Executing neura.grant_once:\n";
  }

  bool hasValue = op.getValue() != nullptr;
  bool hasConstant = op.getConstantValue().has_value();
  
  if (hasValue == hasConstant) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ grant_once requires exactly one of (value operand, constant_value attribute)\n";
    }
    return false;
  }

  PredicatedData source;
  if (hasValue) {
    Value inputValue = op.getValue();
    if (!valueMap.count(inputValue)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Source value not found in valueMap\n";
      }
      return false;
    }
    source = valueMap[inputValue];
    
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Source value: " << source.value << ", [pred = " << source.predicate << "]\n";
    }
  } else {
    Attribute constantAttr = op.getConstantValue().value();
    // 使用全局的mlir::dyn_cast替代成员函数dyn_cast
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constantAttr)) {
      source.value = intAttr.getInt(); 
      source.predicate = false; 
      source.isVector = false;
    // 使用全局的mlir::dyn_cast替代成员函数dyn_cast
    } else if (auto floatAttr = mlir::dyn_cast<FloatAttr>(constantAttr)) {
      source.value = floatAttr.getValueAsDouble();
      source.predicate = false;
      source.isVector = false;
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported constant_value type\n";
      }
      return false;
    }
    
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Constant value: " << source.value << "\n";
    }
  }

  static llvm::DenseMap<Operation*, bool> granted;
  bool hasGranted = granted[op.getOperation()];
  bool resultPredicate = !hasGranted;

  if (!hasGranted) {
    granted[op.getOperation()] = true;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ First access - granting predicate\n";
    }
  } else {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Subsequent access - denying predicate\n";
    }
  }

  PredicatedData result = source;
  result.predicate = resultPredicate;
  result.isVector = source.isVector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value 
                 << ", [pred = " << resultPredicate << "]\n";
  }

  valueMap[op.getResult()] = result;
  return true;
}

bool handleGrantAlwaysOp(neura::GrantAlwaysOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.grant_always:\n";
  }

  if (op.getOperation()->getNumOperands() != 1) {
    if (isVerboseMode()) {  
      llvm::errs() << "[neura-interpreter]  └─ neura.grant_always expects exactly 1 operand (value)\n";
    }
    return false;
  }

  auto source = valueMap[op.getValue()];
  bool resultPredicate = true;
  PredicatedData result = source;
  result.predicate = resultPredicate;
  result.isVector = source.isVector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Source value: " << source.value << ", [pred = " << source.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  ├─ Granting predicate unconditionally\n";
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value 
               << ", [pred = " << resultPredicate << "]\n";
  }
  

  valueMap[op.getResult()] = result;
  return true;
}

bool isDataUpdated(const PredicatedData& oldData, const PredicatedData& newData) {
  if (oldData.value != newData.value) return true;
  if (oldData.predicate != newData.predicate) return true;
  if (oldData.isVector != newData.isVector) return true;
  if (oldData.isVector && oldData.vectorData != newData.vectorData) return true;
  return false;
}

void executeOperation(Operation* op, 
                     llvm::DenseMap<Value, PredicatedData>& valueMap, 
                     Memory& mem,
                     llvm::SmallVector<Operation*, 16>& nextWorklist) {
  llvm::DenseMap<Value, PredicatedData> oldValues;
  for (Value result : op->getResults()) {
    if (valueMap.count(result)) {
      oldValues[result] = valueMap[result];
    }
  }

  bool executionSuccess = true;
  if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
    executionSuccess = handleArithConstantOp(constOp, valueMap);
  } else if (auto constOp = dyn_cast<neura::ConstantOp>(op)) {
    executionSuccess = handleNeuraConstantOp(constOp, valueMap);
  } else if (auto movOp = dyn_cast<neura::DataMovOp>(op)) {
    valueMap[movOp.getResult()] = valueMap[movOp.getOperand()];
  } else if (auto addOp = dyn_cast<neura::AddOp>(op)) {
    executionSuccess = handleAddOp(addOp, valueMap);
  } else if (auto subOp = dyn_cast<neura::SubOp>(op)) {
    executionSuccess = handleSubOp(subOp, valueMap);
  } else if (auto faddOp = dyn_cast<neura::FAddOp>(op)) {
    executionSuccess = handleFAddOp(faddOp, valueMap);
  } else if (auto fsubOp = dyn_cast<neura::FSubOp>(op)) {
    executionSuccess = handleFSubOp(fsubOp, valueMap);
  } else if (auto fmulOp = dyn_cast<neura::FMulOp>(op)) {
    executionSuccess = handleFMulOp(fmulOp, valueMap);
  } else if (auto fdivOp = dyn_cast<neura::FDivOp>(op)) {
    executionSuccess = handleFDivOp(fdivOp, valueMap);
  } else if (auto vfmulOp = dyn_cast<neura::VFMulOp>(op)) {
    executionSuccess = handleVFMulOp(vfmulOp, valueMap);
  } else if (auto faddFaddOp = dyn_cast<neura::FAddFAddOp>(op)) {
    executionSuccess = handleFAddFAddOp(faddFaddOp, valueMap);
  } else if (auto fmulFaddOp = dyn_cast<neura::FMulFAddOp>(op)) {
    executionSuccess = handleFMulFAddOp(fmulFaddOp, valueMap);
  } else if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
    executionSuccess = handleFuncReturnOp(retOp, valueMap);
  } else if (auto fcmpOp = dyn_cast<neura::FCmpOp>(op)) {
    executionSuccess = handleFCmpOp(fcmpOp, valueMap);
  } else if (auto icmpOp = dyn_cast<neura::ICmpOp>(op)) {
    executionSuccess = handleICmpOp(icmpOp, valueMap);
  } else if (auto orOp = dyn_cast<neura::OrOp>(op)) {
    executionSuccess = handleOrOp(orOp, valueMap);
  } else if (auto notOp = dyn_cast<neura::NotOp>(op)) {
    executionSuccess = handleNotOp(notOp, valueMap);
  } else if (auto selOp = dyn_cast<neura::SelOp>(op)) {
    executionSuccess = handleSelOp(selOp, valueMap);
  } else if (auto castOp = dyn_cast<neura::CastOp>(op)) {
    executionSuccess = handleCastOp(castOp, valueMap);
  } else if (auto loadOp = dyn_cast<neura::LoadOp>(op)) {
    executionSuccess = handleLoadOp(loadOp, valueMap, mem);
  } else if (auto storeOp = dyn_cast<neura::StoreOp>(op)) {
    executionSuccess = handleStoreOp(storeOp, valueMap, mem);
  } else if (auto gepOp = dyn_cast<neura::GEP>(op)) {
    executionSuccess = handleGEPOp(gepOp, valueMap);
  } else if (auto loadIndexOp = dyn_cast<neura::LoadIndexedOp>(op)) {
    executionSuccess = handleLoadIndexedOp(loadIndexOp, valueMap, mem);
  } else if (auto storeIndexOp = dyn_cast<neura::StoreIndexedOp>(op)) {
    executionSuccess = handleStoreIndexedOp(storeIndexOp, valueMap, mem);
  } else if (auto brOp = dyn_cast<neura::Br>(op)) {
    // executionSuccess = handleBrOp(brOp, valueMap, currentBlock, lastVisitedBlock);
  } else if (auto condBrOp = dyn_cast<neura::CondBr>(op)) {
    // executionSuccess = handleCondBrOp(condBrOp, valueMap, currentBlock, lastVisitedBlock);
  } else if (auto phiOp = dyn_cast<neura::PhiOp>(op)) {
    executionSuccess = handlePhiOpDataFlowMode(phiOp, valueMap);
  } else if (auto reserveOp = dyn_cast<neura::ReserveOp>(op)) {
    executionSuccess = handleReserveOp(reserveOp, valueMap);
  } else if (auto ctrlMovOp = dyn_cast<neura::CtrlMovOp>(op)) {
    executionSuccess = handleCtrlMovOpDataFlowMode(ctrlMovOp, valueMap);
  } else if (auto returnOp = dyn_cast<neura::ReturnOp>(op)) {
    executionSuccess = handleNeuraReturnOp(returnOp, valueMap);
  } else if (auto grantPredOp = dyn_cast<neura::GrantPredicateOp>(op)) {
    executionSuccess = handleGrantPredicateOp(grantPredOp, valueMap);
  } else if (auto grantOnceOp = dyn_cast<neura::GrantOnceOp>(op)) {
    executionSuccess = handleGrantOnceOp(grantOnceOp, valueMap);
  } else if (auto grantAlwaysOp = dyn_cast<neura::GrantAlwaysOp>(op)) {
    executionSuccess = handleGrantAlwaysOp(grantAlwaysOp, valueMap);
  } else {
    llvm::errs() << "[neura-interpreter]  Unhandled op: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
    executionSuccess = false;
  }

  if (!executionSuccess) {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Operation failed, no propagation\n";
    }
    return;
  }

  bool hasUpdate = false;
  for (Value result : op->getResults()) {
    if (!valueMap.count(result)) continue;
    PredicatedData newData = valueMap[result];

    if (!oldValues.count(result)) {
      newData.isUpdated = true;
      valueMap[result] = newData; 
      hasUpdate = true;
      continue;
    }

    PredicatedData oldData = oldValues[result];
    if (isDataUpdated(oldData, newData)) {
      newData.isUpdated = true;
      valueMap[result] = newData;
      hasUpdate = true;
    } else {
      newData.isUpdated = false;
      valueMap[result] = newData;
    }
  }

  if (op->getNumResults() == 0) {
    if (auto ctrlMovOp = dyn_cast<neura::CtrlMovOp>(op)) {
      Value target = ctrlMovOp.getTarget();
      if (valueMap.count(target) && valueMap[target].isUpdated) {
        hasUpdate = true;
      } else {
        llvm::outs() << "[neura-interpreter]  No update for ctrl_mov target: " << target << "\n";
      }
    }
  }

  if (hasUpdate) {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Operation updated, propagating to users...\n";
    }

    llvm::SmallVector<Value, 4> affectedValues;
    for (Value result : op->getResults()) {
      if (valueMap.count(result) && valueMap[result].isUpdated) {
        affectedValues.push_back(result);
      }
    }

    if (op->getNumResults() == 0) {
      if (auto ctrlMovOp = dyn_cast<neura::CtrlMovOp>(op)) {
        Value target = ctrlMovOp.getTarget();
        if (valueMap.count(target) && valueMap[target].isUpdated) {
          affectedValues.push_back(target);
        }
      }
    }

    for (Value val : affectedValues) {
      if (!valueUsers.count(val)) continue;
      for (Operation* userOp : valueUsers[val]) {
        if (!inWorklist[userOp]) {
          nextWorklist.push_back(userOp);
          inWorklist[userOp] = true;
          if (isVerboseMode()) {
            llvm::outs() << "[neura-interpreter]  Added user to next worklist: ";
            userOp->print(llvm::outs());
            llvm::outs() << "\n";
          }
        }
      }
    }
  }
}

void initializeStaticOps(func::FuncOp func,
                         llvm::DenseMap<Value, PredicatedData>& valueMap,
                         Memory& mem,
                         llvm::SmallVector<Operation*, 16>& nextWorklist) {
  for (auto& block : func.getBody()) {
    for (auto& op : block.getOperations()) {
      if (isa<neura::ConstantOp>(op) || 
          isa<arith::ConstantOp>(op) ||
          // isa<neura::GrantOnceOp>(op) ||
          isa<neura::ReserveOp>(op)) {
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter] Init phase: executing ";
          op.print(llvm::outs());
          llvm::outs() << "\n";
        }

        executeOperation(&op, valueMap, mem, nextWorklist);
      }
    }
  }
}

int runDataFlowMode_(func::FuncOp func,
                    llvm::DenseMap<Value, PredicatedData>& valueMap,
                    Memory& mem) {

  llvm::SmallVector<Operation*, 16> nextWorklist;
  initializeStaticOps(func, valueMap, mem, nextWorklist);
  worklist.clear();
  inWorklist.clear();
  llvm::SmallVector<Operation*, 4> delayQueue;

  for (auto& block : func.getBody()) {
    for (auto& op : block.getOperations()) {
      if (isa<neura::ReturnOp>(op) || isa<func::ReturnOp>(op)) {
        delayQueue.push_back(&op);
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  Return op added to delay queue: ";
          op.print(llvm::outs());
          llvm::outs() << "\n";
        }
      } else if (!(isa<neura::ConstantOp>(op) || isa<arith::ConstantOp>(op) ||
                  isa<neura::GrantOnceOp>(op) || isa<neura::ReserveOp>(op))) {
        worklist.push_back(&op);
        inWorklist[&op] = true;
      }
    }
  }

  int iterCount = 0;
  while (!worklist.empty() || !delayQueue.empty()) {
    iterCount++;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Iteration " << iterCount 
                   << " | worklist: " << worklist.size() 
                   << " | delayQueue: " << delayQueue.size() << "\n";
    }

    for (Operation* op : worklist) {
      inWorklist[op] = false;
      executeOperation(op, valueMap, mem, nextWorklist);
    }

    bool canExecuteReturn = true;
    if (!delayQueue.empty()) {
      Operation* returnOp = delayQueue[0]; 
      for (Value input : returnOp->getOperands()) {
        if (!valueMap.count(input) || !valueMap[input].predicate) {
          canExecuteReturn = false;
          break;
        }
      }
    }

    if (canExecuteReturn && !delayQueue.empty()) {
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  All return inputs are valid, executing return\n";
      }
      for (Operation* returnOp : delayQueue) {
        nextWorklist.push_back(returnOp);
        inWorklist[returnOp] = true;
      }
      delayQueue.clear();
    } else {
      if (isVerboseMode() && !delayQueue.empty()) {
        llvm::outs() << "[neura-interpreter]  Some return inputs are invalid, keeping return in delay queue\n";
      }
    }
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]" <<
                   " | next work: " << nextWorklist.size() << "\n";
    }
    worklist = std::move(nextWorklist);

  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Total iterations: " << iterCount << "\n";
  }
  return 0;
}

int runDataFlowMode(func::FuncOp func,
                    llvm::DenseMap<Value, PredicatedData>& valueMap,
                    Memory& mem) {
  llvm::SmallVector<Operation*, 4> delayQueue;

  for (auto& block : func.getBody()) {
    for (auto& op : block.getOperations()) {
      if (isa<neura::ReturnOp>(op) || isa<func::ReturnOp>(op)) {
        delayQueue.push_back(&op);
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  Return op added to delay queue: ";
          op.print(llvm::outs());
          llvm::outs() << "\n";
        }
      } else {
        worklist.push_back(&op);
        inWorklist[&op] = true;
      }
    }
  }

  int iterCount = 0;
  while (!worklist.empty() || !delayQueue.empty()) {
    iterCount++;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Iteration " << iterCount
                   << " | worklist: " << worklist.size()
                   << " | delayQueue: " << delayQueue.size() << "\n";
    }

    llvm::SmallVector<Operation*, 16> nextWorklist;

    for (Operation* op : worklist) {
      inWorklist[op] = false;
      executeOperation(op, valueMap, mem, nextWorklist);
    }

    if (!delayQueue.empty()) {
      Operation* returnOp = delayQueue[0];
      bool canExecuteReturn = true;
      for (Value input : returnOp->getOperands()) {
        if (!valueMap.count(input) || !valueMap[input].predicate) {
          canExecuteReturn = false;
          break;
        }
      }

      if (canExecuteReturn) {
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  All return inputs are valid, executing return\n";
        }
        for (Operation* op : delayQueue) {
          nextWorklist.push_back(op);
          inWorklist[op] = true;
        }
        delayQueue.clear();
      } else {
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  Some return inputs are invalid, keeping return in delay queue\n";
        }
      }
    }

    worklist = std::move(nextWorklist);
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Total iterations: " << iterCount << "\n";
  }

  return 0;
}


int runControlFlowMode(func::FuncOp func, 
                      llvm::DenseMap<Value, PredicatedData>& valueMap, 
                      Memory& mem) {
  Block* currentBlock = &func.getBody().front(); 
  Block* lastVisitedBlock = nullptr;             
  size_t opIndex = 0;                            
  bool isTerminated = false;                      

  while (!isTerminated && currentBlock) {
    auto& operations = currentBlock->getOperations();

    if (opIndex >= operations.size()) {
      break;
    }

    Operation& op = *std::next(operations.begin(), opIndex);

    if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
      if (!handleArithConstantOp(constOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto constOp = dyn_cast<neura::ConstantOp>(op)) {
      if (!handleNeuraConstantOp(constOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto movOp = dyn_cast<neura::DataMovOp>(op)) {
      valueMap[movOp.getResult()] = valueMap[movOp.getOperand()];
      ++opIndex;
    } else if (auto addOp = dyn_cast<neura::AddOp>(op)) {
      if (!handleAddOp(addOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto subOp = dyn_cast<neura::SubOp>(op)) {
      if (!handleSubOp(subOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto faddOp = dyn_cast<neura::FAddOp>(op)) {
      if (!handleFAddOp(faddOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto fsubOp = dyn_cast<neura::FSubOp>(op)) {
      if (!handleFSubOp(fsubOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto fmulOp = dyn_cast<neura::FMulOp>(op)) {
      if (!handleFMulOp(fmulOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto fdivOp = dyn_cast<neura::FDivOp>(op)) {
      if (!handleFDivOp(fdivOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto vfmulOp = dyn_cast<neura::VFMulOp>(op)) {
      if (!handleVFMulOp(vfmulOp, valueMap)) return 1;
      opIndex++;
    } else if (auto faddFaddOp = dyn_cast<neura::FAddFAddOp>(op)) {
      if (!handleFAddFAddOp(faddFaddOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto fmulFaddOp = dyn_cast<neura::FMulFAddOp>(op)) {
      if (!handleFMulFAddOp(fmulFaddOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
      if (!handleFuncReturnOp(retOp, valueMap)) return 1;
      isTerminated = true;
      ++opIndex;
    } else if (auto fcmpOp = dyn_cast<neura::FCmpOp>(op)) {
      if (!handleFCmpOp(fcmpOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto icmpOp = dyn_cast<neura::ICmpOp>(op)) {
      if (!handleICmpOp(icmpOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto orOp = dyn_cast<neura::OrOp>(op)) {
      if (!handleOrOp(orOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto notOp = dyn_cast<neura::NotOp>(op)) {
      if (!handleNotOp(notOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto selOp = dyn_cast<neura::SelOp>(op)) {
      if (!handleSelOp(selOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto castOp = dyn_cast<neura::CastOp>(op)) {
      if (!handleCastOp(castOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto loadOp = dyn_cast<neura::LoadOp>(op)) {
      if (!handleLoadOp(loadOp, valueMap, mem)) return 1;
      ++opIndex;
    } else if (auto storeOp = dyn_cast<neura::StoreOp>(op)) {
      if (!handleStoreOp(storeOp, valueMap, mem)) return 1;
      ++opIndex;
    } else if (auto gepOp = dyn_cast<neura::GEP>(op)) {
      if (!handleGEPOp(gepOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto loadIndexOp = dyn_cast<neura::LoadIndexedOp>(op)) {
      if (!handleLoadIndexedOp(loadIndexOp, valueMap, mem)) return 1;
      ++opIndex;
    } else if (auto storeIndexOp = dyn_cast<neura::StoreIndexedOp>(op)) {
      if (!handleStoreIndexedOp(storeIndexOp, valueMap, mem)) return 1;
      ++opIndex;
    } else if (auto brOp = dyn_cast<neura::Br>(op)) {
      if (!handleBrOp(brOp, valueMap, currentBlock, lastVisitedBlock)) return 1;
      opIndex = 0;  
    } else if (auto condBrOp = dyn_cast<neura::CondBr>(op)) {
      if (!handleCondBrOp(condBrOp, valueMap, currentBlock, lastVisitedBlock)) return 1;
      opIndex = 0; 
    } else if (auto phiOp = dyn_cast<neura::PhiOp>(op)) {
      if (!handlePhiOpControlFlowMode(phiOp, valueMap, currentBlock, lastVisitedBlock)) return 1;
      ++opIndex;
    } else if (auto reserveOp = dyn_cast<neura::ReserveOp>(op)) {
      if (!handleReserveOp(reserveOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto ctrlMovOp = dyn_cast<neura::CtrlMovOp>(op)) {
      if (!handleCtrlMovOp(ctrlMovOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto returnOp = dyn_cast<neura::ReturnOp>(op)) {
      if (!handleNeuraReturnOp(returnOp, valueMap)) return 1;
      isTerminated = true;
      ++opIndex;
    } else if (auto grantPredOp = dyn_cast<neura::GrantPredicateOp>(op)) {
      if (!handleGrantPredicateOp(grantPredOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto grantOnceOp = dyn_cast<neura::GrantOnceOp>(op)) {
      if (!handleGrantOnceOp(grantOnceOp, valueMap)) return 1;
      ++opIndex;
    } else if (auto grantAlwaysOp = dyn_cast<neura::GrantAlwaysOp>(op)) {
      if (!handleGrantAlwaysOp(grantAlwaysOp, valueMap)) return 1;
      ++opIndex;
    } else {
      llvm::errs() << "[neura-interpreter]  Unhandled op: ";
      op.print(llvm::errs());
      llvm::errs() << "\n";
      return 1;
    }
  }

  return 0; 
}

int main(int argc, char **argv) {

  for (int i = 0; i < argc; ++i) {
    if (std::string(argv[i]) == "--verbose") {
      setVerboseMode(true);
    } else if (std::string(argv[i]) == "--dataflow") {
      setDataflowMode(true);
    }
  }

  if (argc < 2) {
    llvm::errs() << "[neura-interpreter]  Usage: neura-interpreter <input.mlir> [--verbose]\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<neura::NeuraDialect, func::FuncDialect, arith::ArithDialect>();

  MLIRContext context;
  context.appendDialectRegistry(registry);

  llvm::SourceMgr sourceMgr;
  auto fileOrErr = mlir::openInputFile(argv[1]);
  if (!fileOrErr) {
    llvm::errs() << "[neura-interpreter]  Error opening file\n";
    return 1;
  }

  sourceMgr.AddNewSourceBuffer(std::move(fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "[neura-interpreter]  Failed to parse MLIR input file\n";
    return 1;
  }

  // Changes map to store PredicatedData instead of just float.
  llvm::DenseMap<Value, PredicatedData> valueMap;

  Memory mem(1024); // 1MB

  if (isDataflowMode()) {
    buildDependencyGraph(*module);
    for (auto func : module->getOps<func::FuncOp>()) {
      if (runDataFlowMode(func, valueMap, mem)) {
        llvm::errs() << "[neura-interpreter]  Data Flow execution failed\n";
        return 1;
      }
    }
  } else {
    for (auto func : module->getOps<func::FuncOp>()) {
      if (runControlFlowMode(func, valueMap, mem)) {
        llvm::errs() << "[neura-interpreter]  Control Flow execution failed\n";
        return 1;
      }
    }
  }

  return 0;
}
