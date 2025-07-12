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
};

bool handleArithConstantOp(mlir::arith::ConstantOp op, llvm::DenseMap<Value, PredicatedData>& valueMap) {
  auto attr = op.getValue();
  PredicatedData val{0.0f, true};  // arith constants always have true predicate
      
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    val.value = floatAttr.getValueAsDouble();
  } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    if(intAttr.getType().isInteger(1)) {
      val.value = intAttr.getInt() != 0 ? 1.0f : 0.0f;
    } else {
      val.value = static_cast<float>(intAttr.getInt());
    }
  } else {
    llvm::errs() << "Unsupported constant type in arith.constant\n";
    return false;
  }

  assert(valueMap.count(op.getResult()) == 0 && "Duplicate constant result?");
  valueMap[op.getResult()] = val;
  return true;
}

bool handleNeuraConstantOp(neura::ConstantOp op, llvm::DenseMap<Value, PredicatedData>& valueMap) {
  auto attr = op.getValue();

  llvm::outs() << "[neura-interpreter] Executing neura.constant:\n";
  
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
      llvm::errs() << "Unsupported vector element type in neura.constant\n";
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
            
    llvm::outs() << "  Parsed vector constant of size: " << vectorSize << "\n";
  } else {
    llvm::errs() << "Unsupported constant type in neura.constant\n";
    return false;
  }
  return true;
}

bool handleAddOp(neura::AddOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.add expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  llvm::outs() << "[neura-interpreter] Executing neura.add:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";

  int64_t lhsInt = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhsInt = static_cast<int64_t>(std::round(rhs.value));
  int64_t sum = lhsInt + rhsInt;

  PredicatedData result;
  result.value = static_cast<float>(sum);
  result.predicate = lhs.predicate && rhs.predicate;
  result.isVector = false;

  valueMap[op.getResult()] = result;

  llvm::outs() << "  Result: value = " << result.value << ", predicate = " << result.predicate << "\n";

  return true;
}

bool handleSubOp(neura::SubOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.sub expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getOperand(0)];
  auto rhs = valueMap[op.getOperand(1)];

  llvm::outs() << "[neura-interpreter] Executing neura.sub:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";

  bool finalPredicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
  }

  int64_t lhsInt = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhsInt = static_cast<int64_t>(std::round(rhs.value));
  int64_t resultInt = lhsInt - rhsInt;

  PredicatedData result;
  result.value = static_cast<float>(resultInt);
  result.predicate = finalPredicate;
  result.isVector = false;

  llvm::outs() << "  Result: value = " << result.value << ", predicate = " << result.predicate << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFAddOp(neura::FAddOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.fadd expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  llvm::outs() << "[neura-interpreter] Executing neura.fadd:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";

  PredicatedData result;
  result.value = lhs.value + rhs.value;
  result.predicate = lhs.predicate && rhs.predicate;
  result.isVector = false;

  llvm::outs() << "  Result: value = " << result.value << ", predicate = " << result.predicate << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFSubOp(neura::FSubOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.fsub expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  llvm::outs() << "[neura-interpreter] Executing neura.fsub:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";

  PredicatedData result;
  result.value = lhs.value - rhs.value;
  result.predicate = lhs.predicate && rhs.predicate;
  result.isVector = false;

  llvm::outs() << "  Result: value = " << result.value << ", predicate = " << result.predicate << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFMulOp(neura::FMulOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.fmul expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getOperand(0)];
  auto rhs = valueMap[op.getOperand(1)];

  llvm::outs() << "[neura-interpreter] Executing neura.fmul:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";

  bool finalPredicate = lhs.predicate && rhs.predicate;
  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    llvm::outs() << "  Predicate: value = " << pred.value << ", predicate = " << pred.predicate << "\n";
  }

  float lhsFloat = static_cast<float>(lhs.value);
  float rhsFloat = static_cast<float>(rhs.value);
  float resultFloat = lhsFloat * rhsFloat;

  PredicatedData result;
  result.value = resultFloat;
  result.predicate = finalPredicate;
  result.isVector = false;

  llvm::outs() << "  Result: value = " << result.value << ", predicate = " << result.predicate << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFDivOp(neura::FDivOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.fdiv expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getOperand(0)];
  auto rhs = valueMap[op.getOperand(1)];

  llvm::outs() << "[neura-interpreter] Executing neura.fdiv:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";

  bool finalPredicate = lhs.predicate && rhs.predicate;
  
  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    llvm::outs() << "  Predicate: value = " << pred.value << ", predicate = " << pred.predicate << "\n";
  }

  float resultFloat = 0.0f;
  
  if (finalPredicate) {
    float rhsFloat = static_cast<float>(rhs.value);
    
    if (rhsFloat == 0.0f) {
      resultFloat = std::numeric_limits<float>::quiet_NaN();
      llvm::outs() << "  Warning: Division by zero, result is NaN\n";
    } else {
      float lhsFloat = static_cast<float>(lhs.value);
      resultFloat = lhsFloat / rhsFloat;
    }
  }

  PredicatedData result;
  result.value = resultFloat;
  result.predicate = finalPredicate;
  result.isVector = false;

  llvm::outs() << "  Result: value = " << result.value << ", predicate = " << result.predicate << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleVFMulOp(neura::VFMulOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.vfmul expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  if (!lhs.isVector || !rhs.isVector) {
    llvm::errs() << "[neura-interpreter] neura.vfmul requires both operands to be vectors\n";
    return false;
  }

  if (lhs.vectorData.size() != rhs.vectorData.size()) {
    llvm::errs() << "[neura-interpreter] Vector size mismatch in neura.vfmul\n";
    return false;
  }

  llvm::outs() << "[neura-interpreter] Executing neura.vfmul:\n";
  llvm::outs() << "  LHS: vector size = " << lhs.vectorData.size() 
               << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: vector size = " << rhs.vectorData.size() 
               << ", predicate = " << rhs.predicate << "\n";

  bool finalPredicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto predOperand = valueMap[op.getOperand(2)];
    if (predOperand.isVector) {
      llvm::errs() << "[neura-interpreter] Predicate operand must be a scalar in neura.vfmul\n";
      return false;
    }
    finalPredicate = finalPredicate && (predOperand.value != 0.0f);
    llvm::outs() << "  Predicate: value = " << predOperand.value 
                 << ", predicate = " << predOperand.predicate << "\n";
  }

  PredicatedData result;
  result.isVector = true;
  result.predicate = finalPredicate;
  result.vectorData.resize(lhs.vectorData.size());

  auto printVector = [](ArrayRef<float> vec) {
    llvm::outs() << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      llvm::outs() << vec[i];
      if (i != vec.size() - 1)
        llvm::outs() << ", ";
    }
    llvm::outs() << "]";
  };

  if (finalPredicate) {
    llvm::outs() << "  Vector data:\n";
    llvm::outs() << "    LHS: "; printVector(lhs.vectorData); llvm::outs() << "\n";
    llvm::outs() << "    RHS: "; printVector(rhs.vectorData); llvm::outs() << "\n";
    
    for (size_t i = 0; i < lhs.vectorData.size(); ++i) {
      result.vectorData[i] = lhs.vectorData[i] * rhs.vectorData[i];
    }
    
    llvm::outs() << "    Result: "; printVector(result.vectorData); llvm::outs() << "\n";
  } else {
    std::fill(result.vectorData.begin(), result.vectorData.end(), 0.0f);
    llvm::outs() << "  Predicate is false, setting result vector to zeros\n";
    llvm::outs() << "    Result: [0";
    for (size_t i = 1; i < result.vectorData.size(); ++i)
      llvm::outs() << ", 0";
    llvm::outs() << "]\n";
  }

  llvm::outs() << "  Final result: vector size = " << result.vectorData.size() 
               << ", predicate = " << (finalPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFAddFAddOp(neura::FAddFAddOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 3) {
    llvm::errs() << "[neura-interpreter] neura.fadd_fadd expects at least three operands\n";
    return false;
  }

  auto a = valueMap[op.getA()];
  auto b = valueMap[op.getB()];
  auto c = valueMap[op.getC()];

  llvm::outs() << "[neura-interpreter] Executing neura.fadd_fadd:\n";
  llvm::outs() << "  Operand A: value = " << a.value << ", predicate = " << a.predicate << "\n";
  llvm::outs() << "  Operand B: value = " << b.value << ", predicate = " << b.predicate << "\n";
  llvm::outs() << "  Operand C: value = " << c.value << ", predicate = " << c.predicate << "\n";

  bool finalPredicate = a.predicate && b.predicate && c.predicate;

  if (op.getNumOperands() > 3) {
    auto predOperand = valueMap[op.getOperand(3)];
    finalPredicate = finalPredicate && predOperand.predicate && (predOperand.value != 0.0f);
    llvm::outs() << "  Predicate: value = " << predOperand.value 
                 << ", predicate = " << predOperand.predicate << "\n";
  }

  float resultValue = 0.0f;
  if (finalPredicate) {
    resultValue = (a.value + b.value) + c.value;
    llvm::outs() << "  Calculation: (" << a.value << " + " << b.value << ") + " << c.value 
                 << " = " << resultValue << "\n";
  } else {
    llvm::outs() << "  Predicate is false, result is 0\n";
  }

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = false;

  llvm::outs() << "  Final result: value = " << resultValue 
               << ", predicate = " << (finalPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFMulFAddOp(neura::FMulFAddOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 3) {
    llvm::errs() << "[neura-interpreter] neura.fmul_fadd expects at least three operands\n";
    return false;
  }

  auto a = valueMap[op.getA()];
  auto b = valueMap[op.getB()];
  auto c = valueMap[op.getC()];

  llvm::outs() << "[neura-interpreter] Executing neura.fmul_fadd:\n";
  llvm::outs() << "  Operand A: value = " << a.value << ", predicate = " << a.predicate << "\n";
  llvm::outs() << "  Operand B: value = " << b.value << ", predicate = " << b.predicate << "\n";
  llvm::outs() << "  Operand C: value = " << c.value << ", predicate = " << c.predicate << "\n";

  bool finalPredicate = a.predicate && b.predicate && c.predicate;

  if (op.getNumOperands() > 3) {
    auto predOperand = valueMap[op.getOperand(3)];
    finalPredicate = finalPredicate && predOperand.predicate && (predOperand.value != 0.0f);
    llvm::outs() << "  Predicate: value = " << predOperand.value 
                 << ", predicate = " << predOperand.predicate << "\n";
  }

  float resultValue = 0.0f;
  if (finalPredicate) {
    float mulResult = a.value * b.value;
    resultValue = mulResult + c.value;
    llvm::outs() << "  Calculation: (" << a.value << " * " << b.value << ") + " << c.value 
                 << " = " << mulResult << " + " << c.value << " = " << resultValue << "\n";
  } else {
    llvm::outs() << "  Predicate is false, result is 0\n";
  }

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = false;

  llvm::outs() << "  Final result: value = " << resultValue 
               << ", predicate = " << (finalPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleFuncReturnOp(func::ReturnOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (!op) {
    llvm::errs() << "[neura-interpreter] Expected func.return but got something else\n";
    return false;
  }

  if (op.getNumOperands() == 0) {
    llvm::outs() << "[neura-interpreter] Output: (void)\n";
    return true;
  }

  auto result = valueMap[op.getOperand(0)];
  if (result.isVector) {
          
    llvm::outs() << "[neura-interpreter] Output: ["; 
    for (size_t i = 0; i < result.vectorData.size(); ++i) {
      float val = result.predicate ? result.vectorData[i] : 0.0f;
      llvm::outs() << llvm::format("%.6f", val);
      if (i != result.vectorData.size() - 1)
        llvm::outs() << ", ";
      }
      llvm::outs() << "]\n";
  } else {
    float val = result.predicate ? result.value : 0.0f;
    llvm::outs() << "[neura-interpreter] Output: " << llvm::format("%.6f", val) << "\n";
  }
  return true;
}

bool handleFCmpOp(neura::FCmpOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.fcmp expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  llvm::outs() << "[neura-interpreter] Executing neura.fcmp:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";
  llvm::outs() << "  Comparison type: " << op.getCmpType() << "\n";

  bool pred = true;
  if (op.getNumOperands() > 2) {
    auto predData = valueMap[op.getPredicate()];
    pred = predData.predicate && (predData.value != 0.0f);
    llvm::outs() << "  Predicate operand: value = " << predData.value 
                 << ", predicate = " << predData.predicate << "\n";
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
    llvm::errs() << "[neura-interpreter] Unsupported comparison type: " << cmpType << "\n";
    return false;
  }

  bool finalPredicate = lhs.predicate && rhs.predicate && pred;
  float resultValue = fcmpResult ? 1.0f : 0.0f;

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = false;

  llvm::outs() << "  Comparison result: " << (fcmpResult ? "true" : "false") << "\n";
  llvm::outs() << "  Final result: value = " << resultValue 
               << ", predicate = " << (finalPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleICmpOp(neura::ICmpOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.icmp expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getLhs()];
  auto rhs = valueMap[op.getRhs()];

  llvm::outs() << "[neura-interpreter] Executing neura.icmp:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";
  llvm::outs() << "  Comparison type: " << op.getCmpType() << "\n";

  bool pred = true;
  if (op.getNumOperands() > 2) {
    auto predData = valueMap[op.getPredicate()];
    pred = predData.predicate && (predData.value != 0.0f);
    llvm::outs() << "  Predicate operand: value = " << predData.value 
                 << ", predicate = " << predData.predicate << "\n";
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

  llvm::outs() << "  Signed values: LHS = " << s_lhs << ", RHS = " << s_rhs << "\n";
  llvm::outs() << "  Unsigned values: LHS = " << u_lhs << ", RHS = " << u_rhs << "\n";

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
      llvm::errs() << "[neura-interpreter] Unsupported signed comparison type: " << cmp_type << "\n";
      return false;
    }
  } else if (cmp_type.starts_with("u")) {
    if (cmp_type == "ult") icmp_result = (u_lhs < u_rhs);
    else if (cmp_type == "ule") icmp_result = (u_lhs <= u_rhs);
    else if (cmp_type == "ugt") icmp_result = (u_lhs > u_rhs);
    else if (cmp_type == "uge") icmp_result = (u_lhs >= u_rhs);
    else {
      llvm::errs() << "[neura-interpreter] Unsupported unsigned comparison type: " << cmp_type << "\n";
      return false;
    }
  } else {
    llvm::errs() << "[neura-interpreter] Unsupported comparison type: " << cmp_type << "\n";
    return false;
  }

  bool finalPredicate = lhs.predicate && rhs.predicate && pred;
  float resultValue = icmp_result ? 1.0f : 0.0f;

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = false;

  llvm::outs() << "  Comparison result: " << (icmp_result ? "true" : "false") << "\n";
  llvm::outs() << "  Final result: value = " << resultValue 
               << ", predicate = " << (finalPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleOrOp(neura::OrOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 2) {
    llvm::errs() << "[neura-interpreter] neura.or expects at least two operands\n";
    return false;
  }

  auto lhs = valueMap[op.getOperand(0)];
  auto rhs = valueMap[op.getOperand(1)];

  if (lhs.isVector || rhs.isVector) {
    llvm::errs() << "[neura-interpreter] neura.or requires scalar operands\n";
    return false;
  }

  llvm::outs() << "[neura-interpreter] Executing neura.or:\n";
  llvm::outs() << "  LHS: value = " << lhs.value << ", predicate = " << lhs.predicate << "\n";
  llvm::outs() << "  RHS: value = " << rhs.value << ", predicate = " << rhs.predicate << "\n";

  int64_t lhsInt = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhsInt = static_cast<int64_t>(std::round(rhs.value));
  int64_t resultInt = lhsInt | rhsInt;

  llvm::outs() << "  Bitwise OR: " << lhsInt;
  if (lhsInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
  llvm::outs() << " | " << rhsInt;
  if (rhsInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
  llvm::outs() << " = " << resultInt;
  if (resultInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
  llvm::outs() << "\n";

  bool finalPredicate = lhs.predicate && rhs.predicate;
  if (op.getNumOperands() > 2) {
    auto pred = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && pred.predicate && (pred.value != 0.0f);
    llvm::outs() << "  Predicate operand: value = " << pred.value 
                 << ", predicate = " << pred.predicate << "\n";
  }

  PredicatedData result;
  result.value = static_cast<float>(resultInt);
  result.predicate = finalPredicate;
  result.isVector = false;

  llvm::outs() << "  Final result: value = " << result.value 
               << ", predicate = " << (finalPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleNotOp(neura::NotOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  // if (op.getNumOperands() != 1) {
  //   llvm::errs() << "[neura-interpreter] neura.not expects exactly one operand\n";
  //   return false;
  // }

  auto input = valueMap[op.getOperand()];

  llvm::outs() << "[neura-interpreter] Executing neura.not:\n";
  llvm::outs() << "  Input: value = " << input.value << ", predicate = " << input.predicate << "\n";

  int64_t inputInt = static_cast<int64_t>(std::round(input.value));
  int64_t resultInt = ~inputInt;

  llvm::outs() << "  Bitwise NOT: ~" << inputInt;
  if (inputInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
  llvm::outs() << " = " << resultInt;
  if (resultInt == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
  llvm::outs() << "\n";

  PredicatedData result;
  result.value = static_cast<float>(resultInt);
  result.predicate = input.predicate;
  result.isVector = false;

  llvm::outs() << "  Final result: value = " << result.value 
               << ", predicate = " << (result.predicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleSelOp(neura::SelOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() != 3) {  
    llvm::errs() << "[neura-interpreter] neura.sel expects exactly 3 operands (cond, ifTrue, ifFalse)\n";
    return false;
  }

  auto cond = valueMap[op.getCond()];
  auto ifTrue = valueMap[op.getIfTrue()];
  auto ifFalse = valueMap[op.getIfFalse()];

  llvm::outs() << "[neura-interpreter] Executing neura.sel:\n";
  llvm::outs() << "  Condition: value = " << cond.value << ", predicate = " << cond.predicate << "\n";
  llvm::outs() << "  If true:    value = " << ifTrue.value << ", predicate = " << ifTrue.predicate << "\n";
  llvm::outs() << "  If false:   value = " << ifFalse.value << ", predicate = " << ifFalse.predicate << "\n";

  bool condValue = (cond.value != 0.0f) && cond.predicate;
  llvm::outs() << "  Evaluated condition: " << (condValue ? "true" : "false") << "\n";

  PredicatedData result;
  if (condValue) {
    result.value = ifTrue.value;
    result.predicate = ifTrue.predicate && cond.predicate;  
    llvm::outs() << "  Selecting 'ifTrue' branch\n";
  } else {
    result.value = ifFalse.value;
    result.predicate = ifFalse.predicate && cond.predicate; 
    llvm::outs() << "  Selecting 'ifFalse' branch\n";
  }

  result.isVector = ifTrue.isVector && ifFalse.isVector; 

  llvm::outs() << "  Final result: value = " << result.value 
               << ", predicate = " << (result.predicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleCastOp(neura::CastOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    llvm::errs() << "[neura-interpreter] neura.cast expects 1 or 2 operands\n";
    return false;
  }

  auto input = valueMap[op.getOperand(0)];
  std::string castType = op.getCastType().str();

  llvm::outs() << "[neura-interpreter] Executing neura.cast:\n";
  llvm::outs() << "  Input: value = " << input.value << ", predicate = " << input.predicate << "\n";
  llvm::outs() << "  Cast type: " << castType << "\n";

  bool finalPredicate = input.predicate;
  if (op.getOperation()->getNumOperands() > 1) {
    auto predOperand = valueMap[op.getOperand(1)];
    finalPredicate = finalPredicate && predOperand.predicate && (predOperand.value != 0.0f);
    llvm::outs() << "  Predicate operand: value = " << predOperand.value 
                 << ", predicate = " << predOperand.predicate << "\n";
  }

  float resultValue = 0.0f;
  if (finalPredicate) {
    auto inputType = op.getOperand(0).getType();

    if (castType == "f2i") {
      if (!inputType.isF32()) {
        llvm::errs() << "[neura-interpreter] Cast type 'f2i' requires f32 input\n";
        return false;
      }
      int64_t intValue = static_cast<int64_t>(std::round(input.value));
      resultValue = static_cast<float>(intValue);
      llvm::outs() << "  Converting float to integer: " << input.value << " -> " << intValue << "\n";

    } else if (castType == "i2f") {
      if (!inputType.isInteger()) {
        llvm::errs() << "[neura-interpreter] Cast type 'i2f' requires integer input\n";
        return false;
      }
      int64_t intValue = static_cast<int64_t>(input.value);
      resultValue = static_cast<float>(intValue);
      llvm::outs() << "  Converting integer to float: " << intValue << " -> " << resultValue << "\n";

    } else if (castType == "bool2i" || castType == "bool2f") {
      if (!inputType.isInteger(1)) {
        llvm::errs() << "[neura-interpreter] Cast type '" << castType 
                     << "' requires i1 (boolean) input\n";
        return false;
      }
      bool boolValue = (input.value != 0.0f);
      resultValue = boolValue ? 1.0f : 0.0f;
      llvm::outs() << "  Converting boolean to number: " 
                   << (boolValue ? "true" : "false") << " -> " << resultValue << "\n";

    } else if (castType == "i2bool" || castType == "f2bool") {
      if (!inputType.isInteger() && !inputType.isF32()) {
        llvm::errs() << "[neura-interpreter] Cast type '" << castType 
                     << "' requires integer or f32 input\n";
        return false;
      }
      bool boolValue = (input.value != 0.0f);
      resultValue = boolValue ? 1.0f : 0.0f;
      llvm::outs() << "  Converting number to boolean: " << input.value 
                   << " -> " << (boolValue ? "true" : "false") << " (stored as " << resultValue << ")\n";

    } else {
      llvm::errs() << "[neura-interpreter] Unsupported cast type: " << castType << "\n";
      return false;
    }
  } else {
    llvm::outs() << "  Predicate is false, result is 0\n";
  }

  PredicatedData result;
  result.value = resultValue;
  result.predicate = finalPredicate;
  result.isVector = input.isVector;

  llvm::outs() << "  Final result: value = " << resultValue 
               << ", predicate = " << (finalPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleLoadOp(neura::LoadOp op, llvm::DenseMap<Value, PredicatedData> &valueMap, Memory &mem) {
  auto addrVal = valueMap[op.getOperand(0)];
  bool finalPredicate = addrVal.predicate;

  if (op.getNumOperands() > 1) {
    auto predVal = valueMap[op.getOperand(1)];
    finalPredicate = finalPredicate && predVal.predicate && (predVal.value != 0.0f);
  }

  llvm::outs() << "[neura-interpreter] Executing neura.load:" << "\n";

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
      llvm::errs() << "[neura-interpreter] Unsupported load type\n";
      return false;
    }
  }

  llvm::outs() << "  Load  [addr = " << addr << "] => val = "
               << val << " (predicate=" << (finalPredicate ? "true" : "false") << ")\n";

  valueMap[op.getResult()] = { val, finalPredicate };
  return true;
}

bool handleStoreOp(neura::StoreOp op, llvm::DenseMap<Value, PredicatedData> &valueMap, Memory &mem) {
  auto valData = valueMap[op.getOperand(0)];
  auto addrVal = valueMap[op.getOperand(1)];
  bool finalPredicate = addrVal.predicate;

  if (op.getNumOperands() > 2) {
    auto predVal = valueMap[op.getOperand(2)];
    finalPredicate = finalPredicate && predVal.predicate && (predVal.value != 0.0f);
  }

  llvm::outs() << "[neura-interpreter] Executing neura.store:" << "\n";

  size_t addr = static_cast<size_t>(addrVal.value);
  if (finalPredicate) {
    auto valType = op.getOperand(0).getType();
    if (valType.isF32()) {
      mem.store<float>(addr, valData.value);
    } else if (valType.isInteger(32)) {
      mem.store<int32_t>(addr, static_cast<int32_t>(valData.value));
    } else if (valType.isInteger(1)) {
      mem.store<bool>(addr, (valData.value != 0.0f));
    } else {
      llvm::errs() << "[neura-interpreter] Unsupported store type\n";
      return false;
    }

    llvm::outs() << "  Store [addr = " << addr << "] <= val = "
                 << valData.value << " (predicate=true)\n";

  } else {
    llvm::outs() << "  Store [addr = " << addr
                 << "] skipped due to predicate=false\n";
  }

  return true;
}

bool handleGEPOp(neura::GEP op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getOperation()->getNumOperands() < 1) {
    llvm::errs() << "[neura-interpreter] neura.gep expects at least 1 operand (base address)\n";
    return false;
  }

  auto baseVal = valueMap[op.getOperand(0)];
  size_t baseAddr = static_cast<size_t>(baseVal.value);
  bool finalPredicate = baseVal.predicate;

  llvm::outs() << "[neura-interpreter] Executing neura.gep:\n";
  llvm::outs() << "  Base address: value = " << baseAddr << ", predicate = " << baseVal.predicate << "\n";

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
    llvm::errs() << "[neura-interpreter] neura.gep requires 'strides' attribute\n";
    return false;
  }

  std::vector<size_t> strides;
  for (auto s : stridesAttr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(s);
    if (!intAttr) {
      llvm::errs() << "[neura-interpreter] Invalid type in 'strides' attribute (expected integer)\n";
      return false;
    }
    strides.push_back(static_cast<size_t>(intAttr.getInt()));
  }

  if (indexCount != strides.size()) {
    llvm::errs() << "[neura-interpreter] GEP index count (" << indexCount 
                 << ") mismatch with strides size (" << strides.size() << ")\n";
    return false;
  }

  size_t offset = 0;
  for (unsigned i = 0; i < indexCount; ++i) {
    auto idxVal = valueMap[op.getOperand(i + 1)]; 
    if (!idxVal.predicate) {
      llvm::errs() << "[neura-interpreter] GEP index " << i << " has false predicate\n";
      return false;
    }

    size_t idx = static_cast<size_t>(idxVal.value);
    offset += idx * strides[i];
    llvm::outs() << "  Index " << i << ": value = " << idx << ", stride = " << strides[i] 
                 << ", cumulative offset = " << offset << "\n";
  }

  if (hasPredicate) {
    auto predVal = valueMap[op.getOperand(numOperands - 1)];
    finalPredicate = finalPredicate && predVal.predicate && (predVal.value != 0.0f);
    llvm::outs() << "  Predicate operand: value = " << predVal.value 
                 << ", predicate = " << predVal.predicate << "\n";
  }

  size_t finalAddr = baseAddr + offset;

  PredicatedData result;
  result.value = static_cast<float>(finalAddr);
  result.predicate = finalPredicate;
  result.isVector = false; 

  llvm::outs() << "  Final GEP result: base = " << baseAddr << ", total offset = " << offset 
               << ", final address = " << finalAddr 
               << ", predicate = " << (finalPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleLoadIndexedOp(neura::LoadIndexedOp op,
                         llvm::DenseMap<Value, PredicatedData> &valueMap,
                         Memory &mem) {
  llvm::outs() << "[neura-interpreter] Executing neura.load_indexed:\n";

  auto baseVal = valueMap[op.getBase()];
  if (baseVal.isVector) {
    llvm::errs() << "[neura-interpreter] Vector base not supported in load_indexed\n";
    return false;
  }
  float baseF = baseVal.value;
  bool finalPredicate = baseVal.predicate;

  float offset = 0.0f;
  for (Value idx : op.getIndices()) {
    auto idxVal = valueMap[idx];
    if (idxVal.isVector) {
      llvm::errs() << "[neura-interpreter] Vector index not supported in load_indexed\n";
      return false;
    }
    offset += idxVal.value;
    finalPredicate = finalPredicate && idxVal.predicate;
  }

  if (op.getPredicate()) {
    Value predOperand = op.getPredicate();
    auto predVal = valueMap[predOperand];
    if (predVal.isVector) {
      llvm::errs() << "[neura-interpreter] Vector predicate not supported\n";
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
      llvm::errs() << "[neura-interpreter] Unsupported result type\n";
      return false;
    }
  }

  llvm::outs() << "  LoadIndexed [addr = " << addr << "] => val = "
               << val << " (predicate=" << (finalPredicate ? "true" : "false") << ")\n";

  valueMap[op.getResult()] = { val, finalPredicate, false, {}, false };
  return true;
}

bool handleStoreIndexedOp(neura::StoreIndexedOp op,
                          llvm::DenseMap<Value, PredicatedData> &valueMap,
                          Memory &mem) {
  llvm::outs() << "[neura-interpreter] Executing neura.store_indexed:\n";

  auto valToStore = valueMap[op.getValue()];
  if (valToStore.isVector) {
    llvm::errs() << "[neura-interpreter] Vector value not supported in store_indexed\n";
    return false;
  }
  float value = valToStore.value;
  bool finalPredicate = valToStore.predicate;

  auto baseVal = valueMap[op.getBase()];
  if (baseVal.isVector) {
    llvm::errs() << "[neura-interpreter] Vector base not supported in store_indexed\n";
    return false;
  }
  float baseF = baseVal.value;
  finalPredicate = finalPredicate && baseVal.predicate;

  float offset = 0.0f;
  for (Value idx : op.getIndices()) {
    auto idxVal = valueMap[idx];
    if (idxVal.isVector) {
      llvm::errs() << "[neura-interpreter] Vector index not supported in store_indexed\n";
      return false;
    }
    offset += idxVal.value;
    finalPredicate = finalPredicate && idxVal.predicate;
  }

  if (op.getPredicate()) {
      Value predOperand = op.getPredicate();
      auto predVal = valueMap[predOperand];
      if (predVal.isVector) {
        llvm::errs() << "[neura-interpreter] Vector predicate not supported\n";
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
      llvm::errs() << "[neura-interpreter] Unsupported value type in store_indexed\n";
      return false;
    }
  }

  llvm::outs() << "  StoreIndexed [addr = " << addr << "] <= val = "
               << value << " (predicate=" << (finalPredicate ? "true" : "false") << ")\n";

  return true;
}

bool handleBrOp(neura::Br op, llvm::DenseMap<Value, PredicatedData> &valueMap, 
                Block *&currentBlock, Block *&lastVisitedBlock) {
  Block *destBlock = op.getDest();
  if (!destBlock) {
    llvm::errs() << "[neura-interpreter] neura.br: Target block does not exist\n";
    return false;
  }

  auto parentFunc = destBlock->getParentOp();
  unsigned blockIndex = 0;
  for (auto &block : parentFunc->getRegion(0)) {
    if (&block == destBlock) break;
    blockIndex++;
  }

  llvm::outs() << "[neura-interpreter] Executing neura.br:\n";
  llvm::outs() << "  Target block: index " << blockIndex << "\n";

  const auto &args = op.getArgs();
  const auto &destParams = destBlock->getArguments();

  if (args.size() != destParams.size()) {
    llvm::errs() << "[neura-interpreter] neura.br: Argument count mismatch (passed " 
                 << args.size() << ", target expects " << destParams.size() << ")\n";
    return false;
  }

  for (size_t i = 0; i < args.size(); ++i) {
    Value destParam = destParams[i];
    Value srcArg = args[i];
    
    if (!valueMap.count(srcArg)) {
      llvm::errs() << "[neura-interpreter] neura.br: Argument " << i 
                   << " (source value) not found in value map\n";
      return false;
    }
    
    valueMap[destParam] = valueMap[srcArg];
    llvm::outs() << "  Pass argument " << i << " to block parameter: value = " 
                 << valueMap[srcArg].value << "\n";
  }

  lastVisitedBlock = currentBlock;
  currentBlock = destBlock;
  llvm::outs() << "  Successfully jumped to block (index " << blockIndex << ")\n";
  llvm::outs() << "  Resetting operation index to start of block\n";

  return true;
}

bool handleCondBrOp(neura::CondBr op, llvm::DenseMap<Value, PredicatedData> &valueMap, 
                    Block *&currentBlock, Block *&lastVisitedBlock) {
  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    llvm::errs() << "[neura-interpreter] neura.cond_br expects 1 or 2 operands (condition + optional predicate)\n";
    return false;
  }

  auto condValue = op.getCondition();
  if (!valueMap.count(condValue)) {
    llvm::errs() << "[neura-interpreter] cond_br: condition value not found in valueMap! (SSA name missing)\n";
    return false;
  }
  auto condData = valueMap[op.getCondition()];

  if (!op.getCondition().getType().isInteger(1)) {
    llvm::errs() << "[neura-interpreter] neura.cond_br: condition must be of type i1 (boolean)\n";
    return false;
  }

  llvm::outs() << "[neura-interpreter] Executing neura.cond_br:\n";
  llvm::outs() << "  Condition value: " << condData.value 
               << ", condition validity: " << condData.predicate << "\n";

  bool finalPredicate = condData.predicate;
  if (op.getNumOperands() > 1) {
    auto predData = valueMap[op.getPredicate()];
    finalPredicate = finalPredicate && predData.predicate && (predData.value != 0.0f);
    llvm::outs() << "  Predicate operand: value = " << predData.value 
                 << ", validity = " << predData.predicate << "\n";
  }

  if (!finalPredicate) {
    llvm::errs() << "[neura-interpreter] neura.cond_br: condition or predicate is invalid\n";
    return false;
  }

  bool isTrueBranch = (condData.value != 0.0f);
  Block *targetBlock = isTrueBranch ? op.getTrueDest() : op.getFalseDest();
  const auto &branchArgs = isTrueBranch ? op.getTrueArgs() : op.getFalseArgs();
  const auto &targetParams = targetBlock->getArguments();

  auto parentFunc = targetBlock->getParentOp();
  unsigned blockIndex = 0;
  for (auto &block : parentFunc->getRegion(0)) {
    if (&block == targetBlock) break;
    blockIndex++;
  }
  
  llvm::outs() << "  Condition result: " << (isTrueBranch ? "true" : "false") 
               << " -> targeting block (index " << blockIndex << ")\n";

  if (branchArgs.size() != targetParams.size()) {
    llvm::errs() << "[neura-interpreter] neura.cond_br: argument count mismatch for " 
                 << (isTrueBranch ? "true" : "false") << " branch (expected " 
                 << targetParams.size() << ", got " << branchArgs.size() << ")\n";
    return false;
  }

  for (size_t i = 0; i < branchArgs.size(); ++i) {
    valueMap[targetParams[i]] = valueMap[branchArgs[i]];
    llvm::outs() << "  Pass argument " << i << " to block parameter: value = " 
                 << valueMap[branchArgs[i]].value << "\n";
  }

  lastVisitedBlock = currentBlock;
  currentBlock = targetBlock;
  
  llvm::outs() << "  Successfully jumped to " << (isTrueBranch ? "true" : "false") 
               << " block (index " << blockIndex << ")\n";

  return true;
}

bool handlePhiOp(neura::PhiOp op, llvm::DenseMap<Value, PredicatedData> &valueMap, 
                 Block *currentBlock, Block *lastVisitedBlock) {
  llvm::outs() << "[neura-interpreter] Executing neura.phi:\n";

  auto predecessorsRange = currentBlock->getPredecessors();
  std::vector<Block*> predecessors(predecessorsRange.begin(), predecessorsRange.end());
  size_t predCount = predecessors.size(); 

  if (predCount == 0) {
    llvm::errs() << "[neura-interpreter] neura.phi: Current block has no predecessors\n";
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
    llvm::errs() << "[neura-interpreter] neura.phi: Last visited block is not a predecessor\n";
    return false;
  }

  auto inputs = op.getInputs();
  size_t inputCount = inputs.size();

  if (inputCount != predCount) {
    llvm::errs() << "[neura-interpreter] neura.phi: Input count (" << inputCount 
                 << ") != predecessor count (" << predCount << ")\n";
    return false;
  }

  if (predIndex >= inputCount) {
    llvm::errs() << "[neura-interpreter] neura.phi: Invalid predecessor index (" << predIndex << ")\n";
    return false;
  }

  Value inputVal = inputs[predIndex];
  if (!valueMap.count(inputVal)) {
    llvm::errs() << "[neura-interpreter] neura.phi: Input value not found in value map\n";
    return false;
  }

  PredicatedData inputData = valueMap[inputVal];
  valueMap[op.getResult()] = inputData;

  llvm::outs() << "  Predecessor blocks (" << predCount << "):\n";
  for (size_t i = 0; i < predCount; ++i) {
    llvm::outs() << "    [" << i << "]: " << "Block@" << predecessors[i];  
    if (i == predIndex) {
      llvm::outs() << " (current path)\n";
    } else {
      llvm::outs() << "\n";
    }
  }

  llvm::outs() << "  Result: " << op.getResult() << "\n";
  llvm::outs() << "    Value: " << inputData.value << "\n";
  llvm::outs() << "    Predicate: " << (inputData.predicate ? "true" : "false") << "\n";

  return true;
}

bool handleReserveOp(neura::ReserveOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  llvm::outs() << "[neura-interpreter] Executing neura.reserve:\n";


  PredicatedData placeholder;
  placeholder.value = 0.0f;
  placeholder.predicate = false;
  placeholder.isReserve = true;

  Value result = op.getResult();
  valueMap[result] = placeholder;

  llvm::outs() << "  Created placeholder: " << result << "\n";
  llvm::outs() << "    Initial value: 0.0f\n";
  llvm::outs() << "    Initial predicate: false\n";
  llvm::outs() << "    Type: " << result.getType() << "\n";

  return true;
}

bool handleCtrlMovOp(neura::CtrlMovOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  llvm::outs() << "[neura-interpreter] Executing neura.ctrl_mov:\n";

  Value source = op.getValue();
  Value target = op.getTarget();

  if (!valueMap.count(source)) {
    llvm::errs() << "[neura-interpreter] neura.ctrl_mov: Source value not found in value map\n";
    return false;
  }

  if (!valueMap.count(target) || !valueMap[target].isReserve) {
    llvm::errs() << "[neura-interpreter] neura.ctrl_mov: Target is not a reserve placeholder\n";
    return false;
  }

  const auto &sourceData = valueMap[source];
  auto &targetData = valueMap[target];

  if (source.getType() != target.getType()) {
    llvm::errs() << "[neura-interpreter] neura.ctrl_mov: Type mismatch (source=" 
                 << source.getType() << ", target=" << target.getType() << ")\n";
    return false;
  }

  llvm::outs() << "  Source: " << source << "\n";
  llvm::outs() << "    Value: " << sourceData.value << "\n";
  llvm::outs() << "    Predicate: " << (sourceData.predicate ? "true" : "false") << "\n";
  llvm::outs() << "  Target: " << target << "\n";
  llvm::outs() << "    Old value: " << targetData.value << "\n";
  llvm::outs() << "    Old predicate: " << (targetData.predicate ? "true" : "false") << "\n";

  targetData.value = sourceData.value;
  targetData.predicate = sourceData.predicate;
  targetData.isVector = sourceData.isVector;
  if (sourceData.isVector) {
    targetData.vectorData = sourceData.vectorData;
  }

  llvm::outs() << "  Updated target placeholder:\n";
  llvm::outs() << "    New value: " << targetData.value << "\n";
  llvm::outs() << "    New predicate: " << (targetData.predicate ? "true" : "false") << "\n";

  return true;
}

bool handleNeuraReturnOp(neura::ReturnOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  llvm::outs() << "[neura-interpreter] Executing neura.return:\n";

  std::vector<PredicatedData> returnValues;
  for (Value val : op.getValues()) {
    if (!valueMap.count(val)) {
      llvm::errs() << "[neura-interpreter] neura.return: Return value not found in value map\n";
      return false;
    }
    returnValues.push_back(valueMap[val]);
  }

  llvm::outs() << "  Return values:";
  if (returnValues.empty()) {
    llvm::outs() << " void\n";
  } else {
    llvm::outs() << "\n";
    for (size_t i = 0; i < returnValues.size(); ++i) {
      const auto &data = returnValues[i];
      llvm::outs() << "    [" << i << "]: ";
      
      if (data.isVector) {
        llvm::outs() << "vector=[";
        for (size_t j = 0; j < data.vectorData.size(); ++j) {
          float val = data.predicate ? data.vectorData[j] : 0.0f;
          llvm::outs() << llvm::format("%.6f", val);
          if (j != data.vectorData.size() - 1) llvm::outs() << ", ";
        }
        llvm::outs() << "]";
      } else {
        float val = data.predicate ? data.value : 0.0f;
        llvm::outs() << llvm::format("%.6f", val);
      }
      
      llvm::outs() << " (predicate=" << (data.predicate ? "true" : "false") << ")\n";
    }
  }

  llvm::outs() << "  Execution terminated successfully\n";
  return true;
}

bool handleGrantPredicateOp(neura::GrantPredicateOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getOperation()->getNumOperands() != 2) {
    llvm::errs() << "[neura-interpreter] neura.grant_predicate expects exactly 2 operands (value, new_predicate)\n";
    return false;
  }

  auto source = valueMap[op.getValue()];
  auto newPred = valueMap[op.getPredicate()];

  llvm::outs() << "[neura-interpreter] Executing neura.grant_predicate:\n";
  llvm::outs() << "  Source value: " << source.value << ", source predicate: " << source.predicate << "\n";
  llvm::outs() << "  New predicate: value = " << newPred.value << ", predicate = " << newPred.predicate << "\n";

  bool resultPredicate = source.predicate && newPred.predicate && (newPred.value != 0.0f);

  PredicatedData result = source;
  result.predicate = resultPredicate;
  result.isVector = source.isVector; 

  llvm::outs() << "  Result predicate: " << resultPredicate << "\n";
  llvm::outs() << "  Final result: value = " << result.value 
               << ", predicate = " << (resultPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleGrantOnceOp(neura::GrantOnceOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getOperation()->getNumOperands() != 1) {
    llvm::errs() << "[neura-interpreter] neura.grant_once expects exactly 1 operand (value)\n";
    return false;
  }

  auto source = valueMap[op.getValue()];
  
  llvm::outs() << "[neura-interpreter] Executing neura.grant_once:\n";
  llvm::outs() << "  Source value: " << source.value << ", source predicate: " << source.predicate << "\n";

  static llvm::DenseMap<Value, bool> granted;
  bool hasGranted = granted[op.getValue()]; 

  bool resultPredicate = !hasGranted;
  if (!hasGranted) {
    granted[op.getValue()] = true; 
    llvm::outs() << "  First access - granting predicate\n";
  } else {
    llvm::outs() << "  Subsequent access - denying predicate\n";
  }

  PredicatedData result = source;
  result.predicate = resultPredicate;
  result.isVector = source.isVector; 

  llvm::outs() << "  Result predicate: " << resultPredicate << "\n";
  llvm::outs() << "  Final result: value = " << result.value 
               << ", predicate = " << (resultPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

bool handleGrantAlwaysOp(neura::GrantAlwaysOp op, llvm::DenseMap<Value, PredicatedData> &valueMap) {
  if (op.getOperation()->getNumOperands() != 1) {
    llvm::errs() << "[neura-interpreter] neura.grant_always expects exactly 1 operand (value)\n";
    return false;
  }

  auto source = valueMap[op.getValue()];
  
  llvm::outs() << "[neura-interpreter] Executing neura.grant_always:\n";
  llvm::outs() << "  Source value: " << source.value << ", source predicate: " << source.predicate << "\n";

  bool resultPredicate = true;
  llvm::outs() << "  Granting predicate unconditionally\n";

  PredicatedData result = source;
  result.predicate = resultPredicate;
  result.isVector = source.isVector;

  llvm::outs() << "  Result predicate: " << resultPredicate << "\n";
  llvm::outs() << "  Final result: value = " << result.value 
               << ", predicate = " << (resultPredicate ? "true" : "false") << "\n";

  valueMap[op.getResult()] = result;
  return true;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: neura-interpreter <input.mlir>\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<neura::NeuraDialect, func::FuncDialect, arith::ArithDialect>();

  MLIRContext context;
  context.appendDialectRegistry(registry);

  llvm::SourceMgr sourceMgr;
  auto fileOrErr = mlir::openInputFile(argv[1]);
  if (!fileOrErr) {
    llvm::errs() << "Error opening file\n";
    return 1;
  }

  sourceMgr.AddNewSourceBuffer(std::move(fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse MLIR input file\n";
    return 1;
  }

  // Changes map to store PredicatedData instead of just float.
  llvm::DenseMap<Value, PredicatedData> valueMap;

  Memory mem(1024); // 1MB

  Block *currentBlock = nullptr;
  Block *lastVisitedBlock = nullptr;
  size_t opIndex = 0;
  bool isTerminated = false;

  for (auto func : module->getOps<func::FuncOp>()) {
    currentBlock = &func.getBody().front();
    opIndex = 0;
    isTerminated = false;

    while(!isTerminated && currentBlock) {
      auto& operations = currentBlock->getOperations();
      if(opIndex >= operations.size()) {
        break;
      }

      Operation &op = *std::next(operations.begin(), opIndex);
      if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
        if(!handleArithConstantOp(constOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto constOp = dyn_cast<neura::ConstantOp>(op)) {
        if(!handleNeuraConstantOp(constOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto movOp = dyn_cast<neura::DataMovOp>(op)) {
        valueMap[movOp.getResult()] = valueMap[movOp.getOperand()];
        ++opIndex;
      } else if (auto addOp = dyn_cast<neura::AddOp>(op)) {
        if(!handleAddOp(addOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto subOp = dyn_cast<neura::SubOp>(op)) {
        if(!handleSubOp(subOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto faddOp = dyn_cast<neura::FAddOp>(op)) {
        if(!handleFAddOp(faddOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto fsubOp = dyn_cast<neura::FSubOp>(op)) {
        if(!handleFSubOp(fsubOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto fmulOp = dyn_cast<neura::FMulOp>(op)) {
        if(!handleFMulOp(fmulOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto fdivOp = dyn_cast<neura::FDivOp>(op)) {
        if(!handleFDivOp(fdivOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto vfmulOp = dyn_cast<neura::VFMulOp>(op)) {
        if(!handleVFMulOp(vfmulOp, valueMap)) return 1;
        opIndex++;
      } else if (auto faddFaddOp = dyn_cast<neura::FAddFAddOp>(op)) {
        if(!handleFAddFAddOp(faddFaddOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto fmulFaddOp = dyn_cast<neura::FMulFAddOp>(op)) {
        if(!handleFMulFAddOp(fmulFaddOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
        if(!handleFuncReturnOp(retOp, valueMap)) return 1;
        isTerminated = true;
        ++opIndex;
      } else if (auto fcmpOp = dyn_cast<neura::FCmpOp>(op)) {
        if(!handleFCmpOp(fcmpOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto icmpOp = dyn_cast<neura::ICmpOp>(op)) {
        if(!handleICmpOp(icmpOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto orOp = dyn_cast<neura::OrOp>(op)) {
        if(!handleOrOp(orOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto notOp = dyn_cast<neura::NotOp>(op)) {
        if(!handleNotOp(notOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto selOp = dyn_cast<neura::SelOp>(op)) {
        if(!handleSelOp(selOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto castOp = dyn_cast<neura::CastOp>(op)) {
        if(!handleCastOp(castOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto loadOp = dyn_cast<neura::LoadOp>(op)) {
        if(!handleLoadOp(loadOp, valueMap, mem)) return 1;
        ++opIndex;
      } else if (auto storeOp = dyn_cast<neura::StoreOp>(op)) {
        if(!handleStoreOp(storeOp, valueMap, mem)) return 1;
        ++opIndex;
      } else if (auto gepOp = dyn_cast<neura::GEP>(op)) {
        if(!handleGEPOp(gepOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto loadIndexOp = dyn_cast<neura::LoadIndexedOp>(op)) {
        if(!handleLoadIndexedOp(loadIndexOp, valueMap, mem)) return 1;
        ++opIndex;
      } else if (auto storeIndexOp = dyn_cast<neura::StoreIndexedOp>(op)) {
        if(!handleStoreIndexedOp(storeIndexOp, valueMap, mem)) return 1;
        ++opIndex;
      } else if (auto brOp = dyn_cast<neura::Br>(op)) {
        if(!handleBrOp(brOp, valueMap, currentBlock, lastVisitedBlock)) return 1;
        opIndex = 0;
      } else if (auto condBrOp = dyn_cast<neura::CondBr>(op)) {
        if(!handleCondBrOp(condBrOp, valueMap, currentBlock, lastVisitedBlock)) return 1;
        opIndex = 0;
      } else if (auto phiOp = dyn_cast<neura::PhiOp>(op)) {
        if(!handlePhiOp(phiOp, valueMap, currentBlock, lastVisitedBlock)) return 1;
        ++opIndex;
      } else if (auto reserveOp = dyn_cast<neura::ReserveOp>(op)) {
        if(!handleReserveOp(reserveOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto ctrlMovOp = dyn_cast<neura::CtrlMovOp>(op)) {
        if(!handleCtrlMovOp(ctrlMovOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto returnOp = dyn_cast<neura::ReturnOp>(op)) {
        if(!handleNeuraReturnOp(returnOp, valueMap)) return 1;
        isTerminated = true;
        ++opIndex;
      } else if (auto grantPredOp = dyn_cast<neura::GrantPredicateOp>(op)) {
        if(!handleGrantPredicateOp(grantPredOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto grantOnceOp = dyn_cast<neura::GrantOnceOp>(op)) {
        if(!handleGrantOnceOp(grantOnceOp, valueMap)) return 1;
        ++opIndex;
      } else if (auto grantAlwaysOp = dyn_cast<neura::GrantAlwaysOp>(op)) {
        if(!handleGrantAlwaysOp(grantAlwaysOp, valueMap)) return 1;
        ++opIndex;
      } else {
        llvm::errs() << "Unhandled op: ";
        op.print(llvm::errs());
        llvm::errs() << "\n";
        return 1;
      }
    }
  }

  return 0;
}
