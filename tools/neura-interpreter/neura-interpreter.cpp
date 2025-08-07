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

/**
 * @brief Implements a memory management system with allocation and deallocation capabilities.
 * 
 * This class provides a simulated memory space with malloc/free operations, load/store 
 * operations for data access, and memory visualization. It maintains internal bookkeeping
 * of allocated and free memory blocks using allocation tables and free lists.
 */
class Memory {
public:
    /**
     * @brief Constructs a Memory object with specified size.
     * @param size The total size of memory to allocate (in bytes)
     */
    Memory(size_t size);
    
    /**
     * @brief Loads a value of type T from the specified memory address.
     * @tparam T The type of data to load (must be trivially copyable)
     * @param addr The memory address to load from
     * @return The value loaded from memory
     * @throws std::runtime_error if address is out of bounds
     */
    template <typename T>
    T load(size_t addr) const;
    
    /**
     * @brief Stores a value of type T at the specified memory address.
     * @tparam T The type of data to store (must be trivially copyable)
     * @param addr The memory address to store at
     * @param value The value to store
     * @throws std::runtime_error if address is out of bounds
     */
    template <typename T>
    void store(size_t addr, const T& value);
    
    /**
     * @brief Allocates a contiguous block of memory.
     * @param sizeBytes The size of memory to allocate (in bytes)
     * @return The starting address of the allocated block
     * @throws std::runtime_error if insufficient memory is available
     */
    size_t malloc(size_t sizeBytes);

    /**
     * @brief Deallocates a previously allocated memory block.
     * @param addr The starting address of the block to free
     * @note Silently ignores invalid free operations (prints warning)
     */
    void free(size_t addr);

    /**
     * @brief Dumps memory contents in hexadecimal format.
     * @param start The starting address for the dump (default: 0)
     * @param length The number of bytes to display (default: 64)
     */
    void dump(size_t start = 0, size_t length = 64) const;

    /**
     * @brief Gets the total size of the memory space.
     * @return The total memory size in bytes
     */
    size_t getSize() const;

private:
    std::vector<uint8_t> mem;                         /* The actual memory storage */
    std::unordered_map<size_t, size_t> alloc_table;   /* Tracks allocated blocks (address -> size) */
    std::map<size_t, size_t> free_list;               /* Tracks free blocks (address -> size) */

    /**
     * @brief Validates if a memory access is within bounds.
     * @param addr The starting address to check
     * @param size The size of the memory region to check
     * @return true if access is valid, false otherwise
     */
    bool validAddr(size_t addr, size_t size) const;

    /**
     * @brief Merges adjacent free blocks in the free list.
     * 
     * This internal method coalesces contiguous free blocks to prevent fragmentation
     * and maintain optimal allocation performance.
     */
    void mergeFreeBlocks();
};

Memory::Memory(size_t size) : mem(size, 0) {
    free_list[0] = size;
}

template <typename T>
T Memory::load(size_t addr) const {
    assert(validAddr(addr, sizeof(T)) && "Memory load out of bounds");
    T result;
    std::memcpy(&result, &mem[addr], sizeof(T));
    return result;
}

template <typename T>
void Memory::store(size_t addr, const T& value) {
    assert(validAddr(addr, sizeof(T)) && "Memory store out of bounds");
    std::memcpy(&mem[addr], &value, sizeof(T));
}

size_t Memory::malloc(size_t sizeBytes) {
    for (auto it = free_list.begin(); it != free_list.end(); ++it) {
        if (it->second >= sizeBytes) {
            size_t addr = it->first;
            size_t remain = it->second - sizeBytes;
            free_list.erase(it);
            if (remain > 0) {
                free_list[addr + sizeBytes] = remain;
            }
            alloc_table[addr] = sizeBytes;
            return addr;
        }
    }
    throw std::runtime_error("Out of memory");
}

void Memory::free(size_t addr) {
    auto it = alloc_table.find(addr);
    if (it == alloc_table.end()) {
        std::cerr << "Invalid free at addr " << addr << "\n";
        return;
    }
    size_t size = it->second;
    alloc_table.erase(it);
    free_list[addr] = size;

    mergeFreeBlocks();
}

void Memory::dump(size_t start, size_t length) const {
    for (size_t i = start; i < start + length && i < mem.size(); ++i) {
        printf("%02X ", mem[i]);
        if ((i - start + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

size_t Memory::getSize() const { 
    return mem.size(); 
}

bool Memory::validAddr(size_t addr, size_t size) const {
    return addr + size <= mem.size();
}

void Memory::mergeFreeBlocks() {
    auto it = free_list.begin();
    while (it != free_list.end()) {
        auto curr = it++;
        if (it != free_list.end() && curr->first + curr->second == it->first) {
            curr->second += it->second;
            free_list.erase(it);
            it = curr;
        }
    }
}

// Predicated data structure, used to store scalar/vector values and related metadata
struct PredicatedData {
  float value;                        /* Scalar floating-point value (valid when is_vector is false) */
  bool predicate;                     /* Validity flag: true means the value is valid, false means it should be ignored */
  bool is_vector;                     /* Indicates if it's a vector: true for vector, false for scalar */
  std::vector<float> vector_data;     /* Vector data (valid when is_vector is true) */
  bool is_reserve;                    /* Reserve flag (may be used for memory reservation or temporary storage marking) */
  bool is_updated;                    /* Update flag (indicates whether the data has been modified) */
};

static llvm::DenseMap<Value, llvm::SmallPtrSet<Operation*, 4>> value_users; /* Dependency graph tracking: Maps each value to the set of operations that depend on/use it */
static llvm::SmallVector<Operation*, 16> work_list;                         /* List of operations to process */
static llvm::DenseMap<Operation*, bool> in_work_list;                       /* Marks whether an operation is already in work_list */

static bool verbose = false;          /* Verbose logging mode switch: outputs debug information when true */
static bool dataflow = false;         /* Dataflow analysis mode switch: enables dataflow-related analysis logic when true */

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

/**
 * @brief Builds a dependency graph tracking value-to-user relationships within a module.
 * 
 * This function constructs a graph where each entry maps a Value to the set of Operations
 * that use it as an operand. It skips certain operation types (returns, constants, memory grants)
 * that don't contribute to data flow dependencies.
 * 
 * @param module The MLIR ModuleOp to analyze for value dependencies
 * @return void
 */
void buildDependencyGraph(ModuleOp module) {

  value_users.clear();

  // Traverse all operations in the module
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

    // Record each operand's relationship with the current operation
    for (Value operand : op->getOperands()) {
      value_users[operand].insert(op);
    }
  });

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Dependency Graph:\n";
    for (auto& entry : value_users) {
      llvm::outs() << "[neura-interpreter]  Value: ";
      entry.first.print(llvm::outs());
      llvm::outs() << " -> Users: ";
      for (auto* user_op : entry.second) {
        llvm::outs() << user_op->getName() << ", ";
      }
      llvm::outs() << "\n";
    }
  }
}

/**
 * @brief Adds an operation to the work list if it's not already present.
 * 
 * This function checks if the given operation is valid and not already in the work list
 * before adding it. It also maintains a flag to track presence in the work list for efficiency.
 * Verbose mode will log the addition of operations.
 * 
 * @param op The Operation to be added to the work list.
 * @return void
 */
void addToWorkList(Operation* op) {
  if (op == nullptr)
    return;
  if (in_work_list.lookup(op))
    return;

  work_list.push_back(op);
  in_work_list[op] = true;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter] work_list Added: " << op->getName() << "\n";
  }
}

/**
 * @brief Handles the execution of an arithmetic constant operation (arith.constant) by parsing its value and storing it in the value map.
 * 
 * This function processes MLIR's arith.constant operations, which represent constant values. It extracts the constant value from the operation's
 * attribute, converts it to a floating-point representation (supporting floats, integers, and booleans), and stores it in the value map with a 
 * predicate set to true (since constants are always valid). Unsupported constant types result in an error.
 * 
 * @param op          The arith.constant operation to handle
 * @param value_map   Reference to the map where the parsed constant value will be stored, keyed by the operation's result value
 * @return bool       True if the constant is successfully parsed and stored; false if the constant type is unsupported
 */
bool handleArithConstantOp(mlir::arith::ConstantOp op, llvm::DenseMap<Value, PredicatedData>& value_map) {
  auto attr = op.getValue();
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing arith.constant:\n";
  }

  PredicatedData val{0.0f, true};
  
  // Handle floating-point constants (convert to double-precision float)
  if (auto float_attr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    val.value = float_attr.getValueAsDouble();
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Parsed float constant : " 
                   << llvm::format("%.6f", val.value) << "\n";
    }
  }
  // Handle integer constants (including booleans, which are 1-bit integers) 
  else if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    if(int_attr.getType().isInteger(1)) {
      val.value = int_attr.getInt() != 0 ? 1.0f : 0.0f; 
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  └─ Parsed boolean constant : " 
                     << (val.value ? "true" : "false") << "\n";
      }
    } else {
      val.value = static_cast<float>(int_attr.getInt());
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  └─ Parsed integer constant : " 
                     << llvm::format("%.6f", val.value) << "\n";
      }
    }
  }
  // Handle unsupported constant types  
  else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported constant type in arith.constant\n";
    }
    return false;
  }

  assert(value_map.count(op.getResult()) == 0 && "Duplicate constant result?");
  value_map[op.getResult()] = val;
  return true;
}

/**
 * @brief Handles the execution of a Neura constant operation (neura.constant) by parsing its value (scalar or vector) and storing it in the value map.
 * 
 * This function processes Neura's custom constant operations, which can represent floating-point scalars, integer scalars, or floating-point vectors.
 * It extracts the constant value from the operation's attribute, converts it to the appropriate format, and stores it in the value map. The predicate
 * for the constant can be explicitly set via an attribute (defaulting to true if not specified). Unsupported types or vector element types result in an error.
 * 
 * @param op          The neura.constant operation to handle
 * @param value_map   Reference to the map where the parsed constant value will be stored, keyed by the operation's result value
 * @return bool       True if the constant is successfully parsed and stored; false if the constant type or vector element type is unsupported
 */
bool handleNeuraConstantOp(neura::ConstantOp op, llvm::DenseMap<Value, PredicatedData>& value_map) {
  auto attr = op.getValue();

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.constant:\n";
  }
  // Handle floating-point scalar constants
  if (auto float_attr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    PredicatedData val;
    val.value = float_attr.getValueAsDouble();
    val.predicate = true;
    val.is_vector = false;
            
    if (auto pred_attr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = pred_attr.getValue();
    }
       
    assert(value_map.count(op.getResult()) == 0 && "Duplicate constant result?");
    value_map[op.getResult()] = val;
  }
  // Handle integer scalar constants
  else if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    PredicatedData val;
    val.value = static_cast<float>(int_attr.getInt());
    val.predicate = true;
    val.is_vector = false;
      
    if (auto pred_attr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = pred_attr.getValue();
    }
    
    assert(value_map.count(op.getResult()) == 0 && "Duplicate constant result?");       
    value_map[op.getResult()] = val;
  } 
  // Handle vector constants (dense element attributes)
  else if (auto dense_attr = llvm::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!dense_attr.getElementType().isF32()) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported vector element type in neura.constant\n";
      }
      return false;
    }
            
    PredicatedData val;
    val.is_vector = true;
    val.predicate = true;
          
    size_t vector_size = dense_attr.getNumElements();
    val.vector_data.resize(vector_size);
            
    auto float_values = dense_attr.getValues<float>();
    std::copy(float_values.begin(), float_values.end(), val.vector_data.begin());
            
    if (auto pred_attr = op->getAttrOfType<BoolAttr>("predicate")) {
      val.predicate = pred_attr.getValue();
    }
     
    assert(value_map.count(op.getResult()) == 0 && "Duplicate constant result?");
    value_map[op.getResult()] = val;
    
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Parsed vector constant of size: " << vector_size << "\n";
    }
  }
  // Handle unsupported constant types 
  else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported constant type in neura.constant\n";
    }
    return false;
  }
  return true;
}

/**
 * @brief Handles the execution of a Neura addition operation (neura.add) by computing the sum of integer operands.
 * 
 * This function processes Neura's addition operations, which take 2-3 operands: two integer inputs (LHS and RHS) 
 * and an optional predicate operand. It computes the sum of the integer values, combines the predicates of all 
 * operands (including the optional predicate if present), and stores the result in the value map. The operation 
 * requires at least two operands; fewer will result in an error.
 * 
 * @param op          The neura.add operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the addition is successfully computed; false if there are fewer than 2 operands
 */
bool handleAddOp(neura::AddOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.add:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.add expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getLhs()];
  auto rhs = value_map[op.getRhs()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  int64_t lhs_int = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhs_int = static_cast<int64_t>(std::round(rhs.value));
  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  int64_t sum = lhs_int + rhs_int;

  PredicatedData result;
  result.value = static_cast<float>(sum);
  result.predicate = final_predicate;
  result.is_vector = false;

  value_map[op.getResult()] = result;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura subtraction operation (neura.sub) by computing the difference of integer operands.
 * 
 * This function processes Neura's subtraction operations, which take 2-3 operands: two integer inputs (LHS and RHS)
 * and an optional predicate operand. It computes the difference of the integer values (LHS - RHS), combines the
 * predicates of all operands (including the optional predicate if present), and stores the result in the value map.
 * The operation requires at least two operands; fewer will result in an error.
 * 
 * @param op          The neura.sub operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the subtraction is successfully computed; false if there are fewer than 2 operands
 */
bool handleSubOp(neura::SubOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.sub:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.sub expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getOperand(0)];
  auto rhs = value_map[op.getOperand(1)];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  int64_t lhs_int = static_cast<int64_t>(std::round(lhs.value));
  int64_t rhs_int = static_cast<int64_t>(std::round(rhs.value));
  int64_t result_int = lhs_int - rhs_int;

  PredicatedData result;
  result.value = static_cast<float>(result_int);
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n";
  } 

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point addition operation (neura.fadd) by computing the sum of floating-point operands.
 * 
 * This function processes Neura's floating-point addition operations, which take 2-3 operands: two floating-point inputs (LHS and RHS)
 * and an optional predicate operand. It computes the sum of the floating-point values, combines the predicates of all operands
 * (including the optional predicate if present), and stores the result in the value map. The operation requires at least two operands;
 * fewer will result in an error.
 * 
 * @param op          The neura.fadd operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the floating-point addition is successfully computed; false if there are fewer than 2 operands
 */
bool handleFAddOp(neura::FAddOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fadd:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fadd expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getLhs()];
  auto rhs = value_map[op.getRhs()];
  bool final_predicate = lhs.predicate && rhs.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  if (op.getNumOperands() > 2) {
    auto pred = value_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  PredicatedData result;
  result.value = lhs.value + rhs.value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n"; 
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point subtraction operation (neura.fsub) by computing the difference of floating-point operands.
 * 
 * This function processes Neura's floating-point subtraction operations, which take 2-3 operands: two floating-point inputs (LHS and RHS)
 * and an optional predicate operand. It calculates the difference of the floating-point values (LHS - RHS), combines the predicates of all
 * operands (including the optional predicate if present), and stores the result in the value map. The operation requires at least two operands;
 * fewer will result in an error.
 * 
 * @param op          The neura.fsub operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the floating-point subtraction is successfully computed; false if there are fewer than 2 operands
 */
bool handleFSubOp(neura::FSubOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fsub:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fsub expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getLhs()];
  auto rhs = value_map[op.getRhs()];
  bool final_predicate = lhs.predicate && rhs.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  if (op.getNumOperands() > 2) {
    auto pred = value_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }
  
  PredicatedData result;
  result.value = lhs.value - rhs.value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n"; 
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point multiplication operation (neura.fmul) by computing the product of floating-point operands.
 * 
 * This function processes Neura's floating-point multiplication operations, which take 2-3 operands: two floating-point inputs (LHS and RHS)
 * and an optional predicate operand. It calculates the product of the floating-point values (LHS * RHS), combines the predicates of all
 * operands (including the optional predicate if present), and stores the result in the value map. The operation requires at least two operands;
 * fewer will result in an error.
 * 
 * @param op          The neura.fmul operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the floating-point multiplication is successfully computed; false if there are fewer than 2 operands
 */
bool handleFMulOp(neura::FMulOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fmul:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fmul expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getOperand(0)];
  auto rhs = value_map[op.getOperand(1)];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  float lhs_float = static_cast<float>(lhs.value);
  float rhs_float = static_cast<float>(rhs.value);
  float result_float = lhs_float * rhs_float;

  PredicatedData result;
  result.value = result_float;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point division operation (neura.fdiv) by computing the quotient of floating-point operands.
 * 
 * This function processes Neura's floating-point division operations, which take 2-3 operands: two floating-point inputs (dividend/LHS and divisor/RHS)
 * and an optional predicate operand. It calculates the quotient of the floating-point values (LHS / RHS), handles division by zero by returning NaN,
 * combines the predicates of all operands (including the optional predicate if present), and stores the result in the value map. The operation requires
 * at least two operands; fewer will result in an error.
 * 
 * @param op          The neura.fdiv operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the floating-point division is successfully computed (including division by zero cases); false if there are fewer than 2 operands
 */
bool handleFDivOp(neura::FDivOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fdiv:\n";
  }
  
  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fdiv expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getOperand(0)];
  auto rhs = value_map[op.getOperand(1)];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : value = " << lhs.value << " [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : value = " << rhs.value << " [pred = " << rhs.predicate << "]\n";
  }

  bool final_predicate = lhs.predicate && rhs.predicate;
  
  if (op.getNumOperands() > 2) {
    auto pred = value_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  float result_float = 0.0f;
  float rhs_float = static_cast<float>(rhs.value);

  if (rhs_float == 0.0f) {
    // Return quiet NaN for division by zero to avoid runtime errors
    result_float = std::numeric_limits<float>::quiet_NaN();
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Warning: Division by zero, result is NaN\n";
    }
  } else {
    float lhs_float = static_cast<float>(lhs.value);
    result_float = lhs_float / rhs_float;
  }

  PredicatedData result;
  result.value = result_float;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : value = " << result.value << " [pred = " << result.predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura vector floating-point multiplication operation (neura.vfmul) by computing element-wise products of vector operands.
 * 
 * This function processes Neura's vector floating-point multiplication operations, which take 2-3 operands: two vector inputs (LHS and RHS) 
 * and an optional scalar predicate operand. It validates that both primary operands are vectors of equal size, computes element-wise products, 
 * combines the predicates of all operands (including the optional scalar predicate if present), and stores the resulting vector in the value map. 
 * Errors are returned for invalid operand types (non-vectors), size mismatches, or vector predicates.
 * 
 * @param op          The neura.vfmul operation to handle
 * @param value_map   Reference to the map where the resulting vector will be stored, keyed by the operation's result value
 * @return bool       True if the vector multiplication is successfully computed; false if there are invalid operands, size mismatches, or other errors
 */
bool handleVFMulOp(neura::VFMulOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.vfmul:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.vfmul expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getLhs()];
  auto rhs = value_map[op.getRhs()];

  if (!lhs.is_vector || !rhs.is_vector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.vfmul requires both operands to be vectors\n";
    }
    return false;
  }

  auto print_vector = [](ArrayRef<float> vec) {
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
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS  : vector size = " << lhs.vector_data.size() << ", ";
    print_vector(lhs.vector_data);
    llvm::outs() << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS  : vector size = " << rhs.vector_data.size() << ", ";
    print_vector(rhs.vector_data);
    llvm::outs() << ", [pred = " << rhs.predicate << "]\n";
  }

  if (lhs.vector_data.size() != rhs.vector_data.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Vector size mismatch in neura.vfmul\n";
    }
    return false;
  }

  bool final_predicate = lhs.predicate && rhs.predicate;

  if (op.getNumOperands() > 2) {
    auto pred = value_map[op.getOperand(2)];
    if (pred.is_vector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Predicate operand must be a scalar in neura.vfmul\n";
      }
      return false;
    }
    final_predicate = final_predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred.value << " [pred = " << pred.predicate << "]\n";
    }
  }

  PredicatedData result;
  result.is_vector = true;
  result.predicate = final_predicate;
  result.vector_data.resize(lhs.vector_data.size());
  // Compute element-wise multiplication
  for (size_t i = 0; i < lhs.vector_data.size(); ++i) {
    result.vector_data[i] = lhs.vector_data[i] * rhs.vector_data[i];
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result  : " << "vector size = " << result.vector_data.size() << ", ";
    print_vector(result.vector_data); 
    llvm::outs() << ", [pred = " << result.predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura chained floating-point addition operation (neura.fadd_fadd) by computing a three-operand sum.
 * 
 * This function processes Neura's chained floating-point addition operations, which take 3-4 operands: three floating-point inputs (A, B, C)
 * and an optional predicate operand. It calculates the sum using the order ((A + B) + C), combines the predicates of all operands
 * (including the optional predicate if present), and stores the result in the value map. The operation requires at least three operands;
 * fewer will result in an error.
 * 
 * @param op          The neura.fadd_fadd operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the chained floating-point addition is successfully computed; false if there are fewer than 3 operands
 */
bool handleFAddFAddOp(neura::FAddFAddOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fadd_fadd:\n";
  }

  if (op.getNumOperands() < 3) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fadd_fadd expects at least three operands\n";
    }
    return false;
  }

  auto a = value_map[op.getA()];
  auto b = value_map[op.getB()];
  auto c = value_map[op.getC()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand A : value = " << a.value << ", [pred = " << a.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand B : value = " << b.value << ", [pred = " << b.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Operand C : value = " << c.value << ", [pred = " << c.predicate << "]\n";
  }

  bool final_predicate = a.predicate && b.predicate && c.predicate;

  if (op.getNumOperands() > 3) {
    auto pred_operand = value_map[op.getOperand(3)];
    final_predicate = final_predicate && pred_operand.predicate && (pred_operand.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << pred_operand.value 
                   << " [pred = " << pred_operand.predicate << "]\n";
    }
  }

  // Compute the chained sum: ((A + B) + C)
  float result_value = (a.value + b.value) + c.value;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Calculation  : (" << a.value << " + " << b.value << ") + " << c.value 
                 << " = " << result_value << "\n";
  } 

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result       : value = " << result_value 
                 << ", [pred = " << final_predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura fused multiply-add operation (neura.fmul_fadd) by computing (A * B) + C.
 * 
 * This function processes Neura's fused multiply-add operations, which take 3-4 operands: three floating-point inputs (A, B, C)
 * and an optional predicate operand. It calculates the result using the formula (A * B) + C, combines the predicates of all
 * operands (including the optional predicate if present), and stores the result in the value map. The operation requires at
 * least three operands; fewer will result in an error.
 * 
 * @param op          The neura.fmul_fadd operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the fused multiply-add is successfully computed; false if there are fewer than 3 operands
 */
bool handleFMulFAddOp(neura::FMulFAddOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fmul_fadd:\n";
  }
  if (op.getNumOperands() < 3) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.fmul_fadd expects at least three operands\n";
    }
    return false;
  }

  auto a = value_map[op.getA()];
  auto b = value_map[op.getB()];
  auto c = value_map[op.getC()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand A : value = " << a.value << ", [pred = " << a.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Operand B : value = " << b.value << ", [pred = " << b.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Operand C : value = " << c.value << ", [pred = " << c.predicate << "]\n";
  }

  bool final_predicate = a.predicate && b.predicate && c.predicate;

  if (op.getNumOperands() > 3) {
    auto pred_operand = value_map[op.getOperand(3)];
    final_predicate = final_predicate && pred_operand.predicate && (pred_operand.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << pred_operand.value 
                   << ", [pred = " << pred_operand.predicate << "]\n";
    }
  }
  // Compute the fused multiply-add: (A * B) + C
  float result_value = 0.0f;
  float mul_result = a.value * b.value;
  result_value = mul_result + c.value;
  
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Calculation  : (" << a.value << " * " << b.value << ") + " << c.value 
                 << " = " << mul_result << " + " << c.value << " = " << result_value << "\n";
  }

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result       : value = " << result_value 
                 << ", [pred = " << final_predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a function return operation (func.return) by outputting the return value (if any).
 * 
 * This function processes MLIR's standard function return operations, which may optionally return a single value. 
 * It retrieves the return value from the value map (if present) and prints it in a human-readable format—either as a 
 * scalar or a vector. For vector values, it formats elements as a comma-separated list. If no return value is present 
 * (void return), it indicates a void output.
 * 
 * @param op          The func.return operation to handle
 * @param value_map   Reference to the map storing predicated data for values (used to retrieve the return value)
 * @return bool       True if the return operation is processed successfully; false only if the operation is invalid (nullptr)
 */
bool handleFuncReturnOp(func::ReturnOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
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

  auto result = value_map[op.getOperand(0)];
  // Print vector return value if the result is a vector
  if (result.is_vector) {
    llvm::outs() << "[neura-interpreter]  → Output: ["; 
    for (size_t i = 0; i < result.vector_data.size(); ++i) {
      float val = result.predicate ? result.vector_data[i] : 0.0f;
      llvm::outs() << llvm::format("%.6f", val);
      if (i != result.vector_data.size() - 1)
        llvm::outs() << ", ";
      }
      llvm::outs() << "]\n";
  } else {
    float val = result.predicate ? result.value : 0.0f;
    llvm::outs() << "[neura-interpreter]  → Output: " << llvm::format("%.6f", val) << "\n";
  }
  return true;
}

/**
 * @brief Handles the execution of a Neura floating-point comparison operation (neura.fcmp) by evaluating a specified comparison between two operands.
 * 
 * This function processes Neura's floating-point comparison operations, which take 2-3 operands: two floating-point inputs (LHS and RHS) 
 * and an optional execution predicate. It evaluates the comparison based on the specified type (e.g., "eq" for equality, "lt" for less than), 
 * combines the predicates of all operands (including the optional predicate if present), and stores the result as a boolean scalar (1.0f for true, 0.0f for false) 
 * in the value map. Errors are returned for insufficient operands or unsupported comparison types.
 * 
 * @param op          The neura.fcmp operation to handle
 * @param value_map   Reference to the map where the comparison result will be stored, keyed by the operation's result value
 * @return bool       True if the comparison is successfully evaluated; false if there are insufficient operands or an unsupported comparison type
 */
bool handleFCmpOp(neura::FCmpOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.fcmp:\n";
  }
  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {  
      llvm::errs() << "[neura-interpreter]  └─ neura.fcmp expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getLhs()];
  auto rhs = value_map[op.getRhs()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS               : value = " 
               << lhs.value << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS               : value = " 
               << rhs.value << ", [pred = " << rhs.predicate << "]\n";
  }

  bool pred = true;
  if (op.getNumOperands() > 2) {
    auto pred_data = value_map[op.getPredicate()];
    pred = pred_data.predicate && (pred_data.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = " << pred_data.value 
                 << ", [pred = " << pred_data.predicate << "]\n";
    }
  }

  bool fcmp_result = false;
  StringRef cmp_type = op.getCmpType();
  // Evaluate the comparison based on the specified type
  if (cmp_type == "eq") {
    fcmp_result = (lhs.value == rhs.value);
  } else if (cmp_type == "ne") {
    fcmp_result = (lhs.value != rhs.value);
  } else if (cmp_type == "le") {
    fcmp_result = (lhs.value <= rhs.value);
  } else if (cmp_type == "lt") {
    fcmp_result = (lhs.value < rhs.value);
  } else if (cmp_type == "ge") {
    fcmp_result = (lhs.value >= rhs.value);
  } else if (cmp_type == "gt") {
    fcmp_result = (lhs.value > rhs.value);
  } else {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Unsupported comparison type: " << cmp_type << "\n";
    }
    return false;
  }

  bool final_predicate = lhs.predicate && rhs.predicate && pred;
  float result_value = fcmp_result ? 1.0f : 0.0f;

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Comparison type   : " << op.getCmpType() << "\n";  
    llvm::outs() << "[neura-interpreter]  │  └─ Comparison result : " 
                 << (fcmp_result ? "true" : "false") << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result               : value = " 
                 << result_value << ", [pred = " << final_predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura integer comparison operation (neura.icmp) by evaluating signed/unsigned comparisons between integer operands.
 * 
 * This function processes Neura's integer comparison operations, which take 2-3 operands: two integer inputs (LHS and RHS, stored as floats)
 * and an optional execution predicate. It converts the floating-point stored values to integers, evaluates the comparison based on the specified
 * type (e.g., "eq" for equality, "slt" for signed less than, "ult" for unsigned less than), combines the predicates of all operands, and stores
 * the result as a boolean scalar (1.0f for true, 0.0f for false) in the value map. Errors are returned for insufficient operands or unsupported
 * comparison types.
 * 
 * @param op          The neura.icmp operation to handle
 * @param value_map   Reference to the map where the comparison result will be stored, keyed by the operation's result value
 * @return bool       True if the comparison is successfully evaluated; false if there are insufficient operands or an unsupported comparison type
 */
bool handleICmpOp(neura::ICmpOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.icmp:\n";
  }
  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.icmp expects at least two operands\n";
    }
    return false;
  }

  auto lhs = value_map[op.getLhs()];
  auto rhs = value_map[op.getRhs()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS               : value = " << lhs.value 
                 << ", [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS               : value = " << rhs.value 
                 << ", [pred = " << rhs.predicate << "]\n";
  }

  bool pred = true;
  if (op.getNumOperands() > 2) {
    auto pred_data = value_map[op.getPredicate()];
    pred = pred_data.predicate && (pred_data.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = " << pred_data.value 
                 << ", [pred = " << pred_data.predicate << "]\n";
    }
  }
  // Convert stored floating-point values to signed integers (rounded to nearest integer)
  int64_t s_lhs = static_cast<int64_t>(std::round(lhs.value));
  int64_t s_rhs = static_cast<int64_t>(std::round(rhs.value));

  auto signed_to_unsigned = [](int64_t val) {
    return val >= 0 ? 
           static_cast<uint64_t>(val) : 
           static_cast<uint64_t>(UINT64_MAX + val + 1);
  };
  // Convert signed integers to unsigned for unsigned comparisons
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
  // Evaluate the comparison based on the specified type (signed, unsigned, or equality)
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
  }
  // Handle unsigned comparisons 
  else if (cmp_type.starts_with("u")) {
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

  bool final_predicate = lhs.predicate && rhs.predicate && pred;
  float result_value = icmp_result ? 1.0f : 0.0f;

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  │  └─ Comparison result : " << (icmp_result ? "true" : "false") << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result               : value = " << result_value 
                 << ", [pred = " << final_predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

// bool handleOrOp(neura::OrOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
//   if (isVerboseMode()) {
//     llvm::outs() << "[neura-interpreter]  Executing neura.or:\n";
//   }

//   if (op.getNumOperands() < 2) {
//     if (isVerboseMode()) {
//       llvm::errs() << "[neura-interpreter]  └─ neura.or expects at least two operands\n";
//     }
//     return false;
//   }

//   auto lhs = value_map[op.getOperand(0)];
//   auto rhs = value_map[op.getOperand(1)];

//   if (lhs.is_vector || rhs.is_vector) {
//     if (isVerboseMode()) {
//       llvm::errs() << "[neura-interpreter]  └─ neura.or requires scalar operands\n";
//     }
//     return false;
//   }

//   if (isVerboseMode()) {
//     llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
//     llvm::outs() << "[neura-interpreter]  │  ├─ LHS        : value = " << lhs.value << ", [pred = " << lhs.predicate << "]\n";
//     llvm::outs() << "[neura-interpreter]  │  └─ RHS        : value = " << rhs.value << ", [pred = " << rhs.predicate << "]\n";
//   }

//   int64_t lhs_int = static_cast<int64_t>(std::round(lhs.value));
//   int64_t rhs_int = static_cast<int64_t>(std::round(rhs.value));
//   int64_t result_int = lhs_int | rhs_int;

//   bool final_predicate = lhs.predicate && rhs.predicate;
//   if (op.getNumOperands() > 2) {
//     auto pred = value_map[op.getOperand(2)];
//     final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
//     llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
//     llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = " << pred.value 
//                  << ", [pred = " << pred.predicate << "]\n";
//   }

//   if (isVerboseMode()) {
//     llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
//     llvm::outs() << "[neura-interpreter]  │  └─ Bitwise OR : " << lhs_int;
//     if (lhs_int == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
//     llvm::outs() << " | " << rhs_int;
//     if (rhs_int == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
//     llvm::outs() << " = " << result_int;
//     if (result_int == -1) llvm::outs() << " (0xFFFFFFFFFFFFFFFF)";
//     llvm::outs() << "\n";
//   }

//   PredicatedData result;
//   result.value = static_cast<float>(result_int);
//   result.predicate = final_predicate;
//   result.is_vector = false;

//   if (isVerboseMode()) {
//     llvm::outs() << "[neura-interpreter]  └─ Result     : value = " << result.value 
//                  << ", [pred = " << final_predicate << "]\n";
//   }

//   value_map[op.getResult()] = result;
//   return true;
// }

/**
 * @brief Handles the execution of a Neura logical OR operation (neura.or) for scalar boolean values.
 * 
 * This function processes Neura's logical OR operations, which compute the logical OR of two scalar boolean operands.
 * Logical OR returns true if at least one of the operands is true (non-zero). It supports an optional third operand
 * as a predicate to further validate the result. The operation requires scalar operands (no vectors) and returns
 * a scalar result with a combined validity predicate.
 * 
 * @param op          The neura.or operation to handle (modified for logical OR)
 * @param value_map   Reference to the map storing operands and where the result will be stored
 * @return bool       True if the logical OR is successfully executed; false for invalid operands (e.g., vectors, insufficient count)
 */
bool handleOrOp(neura::OrOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.or (logical OR):\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.or (logical) expects at least two operands\n";
    }
    return false;
  }

  // Retrieve left and right operands
  auto lhs = value_map[op.getOperand(0)];
  auto rhs = value_map[op.getOperand(1)];

  if (lhs.is_vector || rhs.is_vector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.or (logical) requires scalar operands\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ LHS        : value = " << lhs.value 
                 << " (boolean: " << (lhs.value != 0.0f ? "true" : "false") << "), [pred = " << lhs.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ RHS        : value = " << rhs.value 
                 << " (boolean: " << (rhs.value != 0.0f ? "true" : "false") << "), [pred = " << rhs.predicate << "]\n";
  }

  // Convert operands to boolean (non-zero = true)
  bool lhs_bool = (lhs.value != 0.0f);
  bool rhs_bool = (rhs.value != 0.0f);
  // Logical OR result: true if either operand is true
  bool result_bool = lhs_bool || rhs_bool;

  // Compute final validity predicate (combines operand predicates and optional predicate)
  bool final_predicate = lhs.predicate && rhs.predicate;
  if (op.getNumOperands() > 2) {
    auto pred = value_map[op.getOperand(2)];
    final_predicate = final_predicate && pred.predicate && (pred.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred           : value = " << pred.value 
                   << ", [pred = " << pred.predicate << "]\n";
    }
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Logical OR : " 
                 << lhs_bool << " || " << rhs_bool << " = " << result_bool << "\n";
  }

  PredicatedData result;
  result.value = result_bool ? 1.0f : 0.0f;
  result.predicate = final_predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result     : value = " << result.value 
                 << " (boolean: " << (result_bool ? "true" : "false") << "), [pred = " << final_predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura logical NOT operation (neura.not) by computing the inverse of a boolean input.
 * 
 * This function processes Neura's logical NOT operations, which take a single boolean operand (represented as a floating-point value). 
 * It converts the input value to an integer (rounded to the nearest whole number), applies the logical NOT operation (inverting true/false), 
 * and stores the result as a floating-point value (1.0f for true, 0.0f for false) in the value map. The result's predicate is inherited 
 * from the input operand's predicate.
 * 
 * @param op          The neura.not operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       Always returns true as the operation is guaranteed to execute successfully with valid input
 */
bool handleNotOp(neura::NotOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  auto input = value_map[op.getOperand()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.not:\n";
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Input       : value = " << input.value 
                 << ", [pred = " << input.predicate << "]\n";
  }

  // Convert the input floating-point value to an integer (rounded to nearest whole number)
  int64_t inputInt = static_cast<int64_t>(std::round(input.value));
  // Apply logical NOT: 0 (false) becomes 1 (true), non-zero (true) becomes 0 (false)
  int64_t result_int = !inputInt;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Logical NOT : !" << inputInt;
    llvm::outs() << "\n";
  }

  PredicatedData result;
  result.value = static_cast<float>(result_int);
  result.predicate = input.predicate;
  result.is_vector = false;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result         : value = " << result.value 
                 << ", [pred = " << result.predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura selection operation (neura.sel) by choosing between two values based on a condition.
 * 
 * This function processes Neura's selection operations, which take exactly 3 operands: a condition, a value to use if the condition is true,
 * and a value to use if the condition is false. It evaluates the condition (treating non-zero values with a true predicate as true),
 * selects the corresponding value (either "if_true" or "if_false"), and combines the predicate of the condition with the predicate of the selected value.
 * The result is marked as a vector only if both input values are vectors. Errors are returned if the operand count is not exactly 3.
 * 
 * @param op          The neura.sel operation to handle
 * @param value_map   Reference to the map where the selected result will be stored, keyed by the operation's result value
 * @return bool       True if the selection is successfully computed; false if the operand count is invalid
 */
bool handleSelOp(neura::SelOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.sel:\n";
  }

  if (op.getNumOperands() != 3) {  
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.sel expects exactly 3 operands (cond, if_true, if_false)\n";
    }
    return false;
  }

  auto cond = value_map[op.getCond()];              /* Condition to evaluate */
  auto if_true = value_map[op.getIfTrue()];         /* Value if condition is true */
  auto if_false = value_map[op.getIfFalse()];       /* Value if condition is false */
  // Evaluate the condition: true if the value is non-zero and its predicate is true
  bool cond_value = (cond.value != 0.0f) && cond.predicate;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operands \n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Condition : value = " << cond.value 
                 << ", [pred = " << cond.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ If true   : value = " << if_true.value 
                 << ", [pred = " << if_true.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ If false  : value = " << if_false.value 
                 << ", [pred = " << if_false.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation \n"; 
  }

  PredicatedData result;
  // Prepare the result by selecting the appropriate value based on the condition
  if (cond_value) {
    result.value = if_true.value;
    result.predicate = if_true.predicate && cond.predicate;  
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Condition is true, selecting 'if_true' branch\n";
    }
  } else {
    result.value = if_false.value;
    result.predicate = if_false.predicate && cond.predicate; 
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  │  └─ Condition is false, selecting 'if_false' branch\n";
    }
  }

  result.is_vector = if_true.is_vector && if_false.is_vector; 

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result      : value = " << result.value 
                 << ", predicate = " << result.predicate << "\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura type conversion operation (neura.cast) by converting an input value between supported types.
 * 
 * This function processes Neura's type conversion operations, which take 1-2 operands: an input value to convert, and an optional predicate operand. 
 * It supports multiple conversion types (e.g., float to integer, integer to boolean) and validates that the input type matches the conversion requirements. 
 * The result's predicate is combined with the optional predicate operand (if present), and the result inherits the input's vector flag. Errors are returned for 
 * invalid operand counts, unsupported conversion types, or mismatched input types for the specified conversion.
 * 
 * @param op          The neura.cast operation to handle
 * @param value_map   Reference to the map where the converted result will be stored, keyed by the operation's result value
 * @return bool       True if the conversion is successfully computed; false for invalid operands, unsupported types, or mismatched input types
 */
bool handleCastOp(neura::CastOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.cast:\n";
  }
  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cast expects 1 or 2 operands\n";
    }
    return false;
  }

  auto input = value_map[op.getOperand(0)];
  std::string cast_type = op.getCastType().str();

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Input  : value = " 
                 << input.value << ", [pred = " << input.predicate << "]\n";
  }

  bool final_predicate = input.predicate;
  if (op.getOperation()->getNumOperands() > 1) {
    auto pred_operand = value_map[op.getOperand(1)];
    final_predicate = final_predicate && pred_operand.predicate && (pred_operand.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n"; 
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << pred_operand.value 
                   << ", [pred = " << pred_operand.predicate << "]\n";
    }
  }
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Cast type : " << cast_type << "\n";
  } 

  float result_value = 0.0f;
  auto input_type = op.getOperand(0).getType();
  // Handle specific conversion types with input type validation
    if (cast_type == "f2i") {
      if (!input_type.isF32()) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Cast type 'f2i' requires f32 input\n";
        }
        return false;
      }
      int64_t int_value = static_cast<int64_t>(std::round(input.value));
      result_value = static_cast<float>(int_value);
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ Converting float to integer " 
        << input.value << " -> " << int_value << "\n";
      }

    } else if (cast_type == "i2f") {
      if (!input_type.isInteger()) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Cast type 'i2f' requires integer input\n";
        }
        return false;
      }
      int64_t int_value = static_cast<int64_t>(input.value);
      result_value = static_cast<float>(int_value);
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ Converting integer to float " 
                     << int_value << " -> " << result_value << "\n";
      }
    } else if (cast_type == "bool2i" || cast_type == "bool2f") {
      if (!input_type.isInteger(1)) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Cast type '" << cast_type 
                       << "' requires i1 (boolean) input\n";
        }
        return false;
      }
      bool bool_value = (input.value != 0.0f);
      result_value = bool_value ? 1.0f : 0.0f;
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ Converting boolean to number " 
                     << (bool_value ? "true" : "false") << " -> " << result_value << "\n";
      }
    } else if (cast_type == "i2bool" || cast_type == "f2bool") {
      if (!input_type.isInteger() && !input_type.isF32()) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Cast type '" << cast_type 
                       << "' requires integer or f32 input\n";
        }
        return false;
      }
      bool bool_value = (input.value != 0.0f);
      result_value = bool_value ? 1.0f : 0.0f;
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ Converting number to boolean " 
                    << input.value << " -> " << (bool_value ? "true" : "false") << " (stored as " << result_value << ")\n";
      }
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported cast type: " << cast_type << "\n";
      }
      return false;
    }

  PredicatedData result;
  result.value = result_value;
  result.predicate = final_predicate;
  result.is_vector = input.is_vector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result    : value = " << result_value 
                 << ", [pred = " << final_predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura load operation (neura.load) by reading a value from memory at a specified address.
 * 
 * This function processes Neura's memory load operations, which take 1-2 operands: a memory address (stored as a float) and an optional predicate operand. 
 * It reads the value from memory at the specified address, with support for 32-bit floats, 32-bit integers, and booleans (1-bit integers). The operation 
 * is skipped if the combined predicate (input address predicate + optional predicate operand) is false, returning a default value of 0.0f. Errors are 
 * returned for invalid operand counts or unsupported data types.
 * 
 * @param op          The neura.load operation to handle
 * @param value_map   Reference to the map where the loaded value will be stored, keyed by the operation's result value
 * @param mem         Reference to the memory object used to read the value
 * @return bool       True if the load is successfully executed (including skipped loads); false for invalid operands or unsupported types
 */
bool handleLoadOp(neura::LoadOp op, llvm::DenseMap<Value, PredicatedData> &value_map, Memory &mem) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.load:\n";
  }

  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.load expects 1 or 2 operands (address, [predicate])\n";
    }
    return false;
  }
  // Convert address from float to size_t (memory address type)
  auto addr_val = value_map[op.getOperand(0)];
  bool final_predicate = addr_val.predicate;

  if (op.getNumOperands() > 1) {
    auto pred_val = value_map[op.getOperand(1)];
    final_predicate = final_predicate && pred_val.predicate && (pred_val.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n"; 
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << pred_val.value 
                   << ", [pred = " << pred_val.predicate << "]\n";
    }
  }

  float val = 0.0f;
  size_t addr = static_cast<size_t>(addr_val.value);
  // Perform load only if final predicate is true
  if (final_predicate) {
    auto result_type = op.getResult().getType();
    if (result_type.isF32()) {
      val = mem.load<float>(addr);
    } else if (result_type.isInteger(32)) {
      val = static_cast<float>(mem.load<int32_t>(addr));
    } else if (result_type.isInteger(1)) {
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
                 << val << ", [pred = " << final_predicate << "]\n";
  }

  value_map[op.getResult()] = { val, final_predicate };
  return true;
}

/**
 * @brief Handles the execution of a Neura store operation (neura.store) by writing a value to memory at a specified address.
 * 
 * This function processes Neura's memory store operations, which take 2-3 operands: a value to store, a memory address (both stored as floats), 
 * and an optional predicate operand. It writes the value to memory at the specified address if the combined predicate (address predicate + 
 * optional predicate operand) is true. Supported types include 32-bit floats, 32-bit integers, and booleans (1-bit integers). The operation 
 * is skipped if the predicate is false. Errors are returned for insufficient operands or unsupported data types.
 * 
 * @param op          The neura.store operation to handle
 * @param value_map   Reference to the map storing the value and address to be used for the store
 * @param mem         Reference to the memory object used to write the value
 * @return bool       True if the store is successfully executed (including skipped stores); false for invalid operands or unsupported types
 */
bool handleStoreOp(neura::StoreOp op, llvm::DenseMap<Value, PredicatedData> &value_map, Memory &mem) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.store:\n";
  }

  if (op.getNumOperands() < 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.store expects at least two operands (value, address)\n";
    }
    return false;
  }

  auto val_data = value_map[op.getOperand(0)];  /* Value to store */
  auto addr_val = value_map[op.getOperand(1)];  /* Target address */
  bool final_predicate = addr_val.predicate;    /* Base predicate from address validity */

  if (op.getNumOperands() > 2) {
    auto pred_val = value_map[op.getOperand(2)];
    final_predicate = final_predicate && pred_val.predicate && (pred_val.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n"; 
      llvm::outs() << "[neura-interpreter]  │  └─ Pred      : value = " << pred_val.value 
                   << ", [pred = " << pred_val.predicate << "]\n";
    }
  }
  // Convert address from float to size_t (memory address type)
  size_t addr = static_cast<size_t>(addr_val.value);
  // Perform store only if final predicate is true
  if(final_predicate) {
    auto val_type = op.getOperand(0).getType();
    if (val_type.isF32()) {
      mem.store<float>(addr, val_data.value);
    } else if (val_type.isInteger(32)) {
      mem.store<int32_t>(addr, static_cast<int32_t>(val_data.value));
    } else if (val_type.isInteger(1)) {
    mem.store<bool>(addr, (val_data.value != 0.0f));
    } else {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Unsupported store type\n";
      }
      return false;
    }
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Store [addr = " << addr 
                   << "] => val = " << val_data.value 
                   << ", [pred = 1" << "]\n";
    }
  } else {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  └─ Store skipped due to [pred = 0]\n";
    }
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura Get Element Pointer operation (neura.gep) by computing a memory address from a base address and indices.
 * 
 * This function processes Neura's GEP operations, which calculate a target memory address by adding an offset to a base address. The offset is computed 
 * using multi-dimensional indices and corresponding strides (step sizes between elements in each dimension). The operation accepts 1 or more operands:
 * a base address, optional indices (one per dimension), and an optional boolean predicate operand (last operand, if present). It requires a "strides" 
 * attribute specifying the stride for each dimension, which determines how much to multiply each index by when calculating the offset.
 * 
 * Key behavior:
 * - Validates operand count (minimum 1: base address)
 * - Identifies optional predicate operand (last operand, if it's a 1-bit integer)
 * - Uses "strides" attribute to determine step sizes for each index dimension
 * - Computes total offset as the sum of (index * stride) for each dimension
 * - Combines predicates from base address and optional predicate operand
 * - Returns the final address (base + offset) with the combined predicate
 * 
 * @param op          The neura.gep operation to handle
 * @param value_map   Reference to the map storing predicated data for values (base address, indices, predicate)
 * @return bool       True if the address is successfully computed; false for invalid operands, missing strides, or mismatched indices/strides
 */
bool handleGEPOp(neura::GEP op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.gep:\n";
  }

  if (op.getOperation()->getNumOperands() < 1) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.gep expects at least 1 operand (base address)\n";
    }
    return false;
  }

  auto base_val = value_map[op.getOperand(0)];
  size_t base_addr = static_cast<size_t>(base_val.value);
  bool final_predicate = base_val.predicate;

  if(isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Base address: value = " << base_addr << ", [pred = " << base_val.predicate << "]\n";
  }

  unsigned num_operands = op.getOperation()->getNumOperands();
  bool has_predicate = false;
  unsigned index_count = num_operands - 1;  

  if (num_operands > 1) {
    auto last_operand_type = op.getOperand(num_operands - 1).getType();
    if (last_operand_type.isInteger(1)) {
      has_predicate = true;
      index_count -= 1; 
    }
  }

  auto strides_attr = op->getAttrOfType<mlir::ArrayAttr>("strides");
  if (!strides_attr) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.gep requires 'strides' attribute\n";
    }
    return false;
  }

  // Convert strides attribute to a vector of size_t (scaling factors for indices)
  std::vector<size_t> strides;
  for (auto s : strides_attr) {
    auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(s);
    if (!int_attr) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Invalid type in 'strides' attribute (expected integer)\n";
      }  
      return false;
    }
    strides.push_back(static_cast<size_t>(int_attr.getInt()));
  }

  if (index_count != strides.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ GEP index count (" << index_count 
                   << ") mismatch with strides size (" << strides.size() << ")\n";
    }
    return false;
  }

  // Calculate total offset by scaling each index with its stride and summing
  size_t offset = 0;
  for (unsigned i = 0; i < index_count; ++i) {
    auto idx_val = value_map[op.getOperand(i + 1)]; 
    if (!idx_val.predicate) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ GEP index " << i << " has false predicate\n";
      }
      return false;
    }

    size_t idx = static_cast<size_t>(idx_val.value);
    offset += idx * strides[i];
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Index " << i << ": value = " << idx << ", stride = " << strides[i] 
                   << ", cumulative offset = " << offset << "\n";
    }
  }

  if (has_predicate) {
    auto pred_val = value_map[op.getOperand(num_operands - 1)];
    final_predicate = final_predicate && pred_val.predicate && (pred_val.value != 0.0f);
    if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Predicate operand: value = " << pred_val.value 
                 << ", [pred = " << pred_val.predicate << "]\n";
    }
  }

  size_t final_addr = base_addr + offset;

  PredicatedData result;
  result.value = static_cast<float>(final_addr);
  result.predicate = final_predicate;
  result.is_vector = false; 

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Final GEP result: base = " << base_addr << ", total offset = " << offset 
                 << ", final address = " << final_addr 
                 << ", [pred = " << final_predicate << "]\n";    
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura indexed load operation (neura.load_indexed) by loading a value from memory using a base address and summed indices.
 * 
 * This function processes Neura's indexed load operations, which calculate a target memory address by adding a base address to the sum of multiple indices. 
 * It supports scalar values only (no vectors) for the base address, indices, and predicate. The operation loads data from the computed address if the combined 
 * predicate (validity of base, indices, and optional predicate operand) is true. Supported data types include 32-bit floats, 32-bit integers, and booleans (1-bit integers).
 * 
 * @param op          The neura.load_indexed operation to handle
 * @param value_map   Reference to the map storing values (base address, indices, predicate) and where the loaded result will be stored
 * @param mem         Reference to the memory object used to read the value
 * @return bool       True if the indexed load is successfully executed (including skipped loads); false for vector inputs, unsupported types, or errors
 */
bool handleLoadIndexedOp(neura::LoadIndexedOp op,
                         llvm::DenseMap<Value, PredicatedData> &value_map,
                         Memory &mem) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Executing neura.load_indexed:\n";
  }

  // Retrieve base address and validate it is not a vector
  auto base_val = value_map[op.getBase()];
  if (base_val.is_vector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Vector base not supported in load_indexed\n";
    }
    return false;
  }

  float base_F = base_val.value;                /* Base address (stored as float) */
  bool final_predicate = base_val.predicate;    /* Initialize predicate with base's validity */

  // Calculate total offset by summing all indices (validate indices are not vectors)
  // Todo: multi-dimensional index will be supported in the future
  float offset = 0.0f;
  for (Value idx : op.getIndices()) {
    auto idx_val = value_map[idx];
    if (idx_val.is_vector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Vector index not supported in load_indexed\n";
      }
      return false;
    }
    // Accumulate index values into total offset
    offset += idx_val.value;
    final_predicate = final_predicate && idx_val.predicate;
  }

  // Incorporate optional predicate operand (validate it is not a vector)
  if (op.getPredicate()) {
    Value pred_operand = op.getPredicate();
    auto pred_val = value_map[pred_operand];
    if (pred_val.is_vector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Vector predicate not supported\n";
      }
      return false;
    }
    final_predicate = final_predicate && pred_val.predicate && (pred_val.value != 0.0f);
  }

  // Compute target address (base + total offset) and initialize loaded value
  size_t addr = static_cast<size_t>(base_F + offset);
  float val = 0.0f;

  // Perform load only if final predicate is true (valid address and conditions)
  if (final_predicate) {
    auto result_type = op.getResult().getType();
    if (result_type.isF32()) {
      val = mem.load<float>(addr);
    } else if (result_type.isInteger(32)) {
      val = static_cast<float>(mem.load<int32_t>(addr));
    } else if (result_type.isInteger(1)) {
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
                 << val << ", [pred = " << final_predicate << "]\n";
  }

  value_map[op.getResult()] = { val, final_predicate, false, {}, false };
  return true;
}

/**
 * @brief Handles the execution of a Neura indexed store operation (neura.store_indexed) by storing a value to memory using a base address and summed indices.
 * 
 * This function processes Neura's indexed store operations, which calculate a target memory address by adding a base address to the sum of multiple indices, then stores a value at that address.
 * It supports scalar values only (no vectors) for the value to store, base address, indices, and predicate. The operation performs the store only if the combined predicate (validity of the value,
 * base, indices, and optional predicate operand) is true. Supported data types include 32-bit floats, 32-bit integers, and booleans (1-bit integers).
 * 
 * @param op          The neura.store_indexed operation to handle
 * @param value_map   Reference to the map storing the value to store, base address, indices, and predicate
 * @param mem         Reference to the memory object used to write the value
 * @return bool       True if the indexed store is successfully executed (including skipped stores); false for vector inputs, unsupported types, or errors
 */
bool handleStoreIndexedOp(neura::StoreIndexedOp op,
                          llvm::DenseMap<Value, PredicatedData> &value_map,
                          Memory &mem) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.store_indexed:\n";
  }

  auto val_to_store = value_map[op.getValue()];
  if (val_to_store.is_vector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Vector value not supported in store_indexed\n";
    }
    return false;
  }
  float value = val_to_store.value;
  bool final_predicate = val_to_store.predicate;

  auto base_val = value_map[op.getBase()];
  if (base_val.is_vector) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Vector base not supported in store_indexed\n";
    }
    return false;
  }

   // Retrieve base address and validate it is not a vector
  float base_F = base_val.value;
  final_predicate = final_predicate && base_val.predicate;

  // Calculate total offset by summing all indices (validate indices are not vectors)
  // Todo: multi-dimensional index will be supported in the future
  float offset = 0.0f;
  for (Value idx : op.getIndices()) {
    auto idx_val = value_map[idx];
    if (idx_val.is_vector) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Vector index not supported in store_indexed\n";
      }
      return false;
    }
    offset += idx_val.value;
    final_predicate = final_predicate && idx_val.predicate;
  }

  if (op.getPredicate()) {
      Value pred_operand = op.getPredicate();
      auto pred_val = value_map[pred_operand];
      if (pred_val.is_vector) {
        if (isVerboseMode()) {
          llvm::errs() << "[neura-interpreter]  └─ Vector predicate not supported\n";
        }
        return false;
      }
      final_predicate = final_predicate && pred_val.predicate && (pred_val.value != 0.0f);
  }

  size_t addr = static_cast<size_t>(base_F + offset);

  if (final_predicate) {
    auto val_type = op.getValue().getType();
    if (val_type.isF32()) {
      mem.store<float>(addr, value);
    } else if (val_type.isInteger(32)) {
      mem.store<int32_t>(addr, static_cast<int32_t>(value));
    } else if (val_type.isInteger(1)) {
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
                 << value << ", [pred = " << final_predicate << "]\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura unconditional branch operation (neura.br) by transferring control to a target block.
 * 
 * This function processes Neura's unconditional branch operations, which unconditionally direct control flow to a specified target block.
 * It validates the target block exists, ensures the number of branch arguments matches the target block's parameters, and copies argument
 * values to the target's parameters. Finally, it updates the current and last visited blocks to reflect the control transfer.
 * 
 * @param op                  The neura.br operation to handle
 * @param value_map           Reference to the map storing branch arguments and where target parameters will be updated
 * @param current_block       Reference to the current block; updated to the target block after branch
 * @param last_visited_block  Reference to the last visited block; updated to the previous current block after branch
 * @return bool               True if the branch is successfully executed; false for invalid target block or argument/parameter mismatch
 */
bool handleBrOp(neura::Br op, llvm::DenseMap<Value, PredicatedData> &value_map, 
                Block *&current_block, Block *&last_visited_block) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.br:\n";
  }

  // Get the target block of the unconditional branch
  Block *dest_block = op.getDest();
  if (!dest_block) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.br: Target block is null\n";
    }
    return false;
  }

  // Retrieve all successor blocks of the current block
  auto current_succs_range = current_block->getSuccessors();
  std::vector<Block *> succ_blocks(current_succs_range.begin(), current_succs_range.end());

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Block Information\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Current block    : Block@" << current_block << "\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Successor blocks : \n";
    for (unsigned int i = 0; i < succ_blocks.size(); ++i) {
    if(i < succ_blocks.size() - 1)
      llvm::outs() << "[neura-interpreter]  │  │  ├─ [" << i << "] Block@" << succ_blocks[i] << "\n";
    else
      llvm::outs() << "[neura-interpreter]  │  │  └─ [" << i << "] Block@" << succ_blocks[i] << "\n";
    }
    llvm::outs() << "[neura-interpreter]  │  └─ Target block : Block@" << dest_block << "\n";
    llvm::outs() << "[neura-interpreter]  ├─ Pass Arguments\n";
  }

  // Get branch arguments and target block parameters
  const auto &args = op.getArgs();
  const auto &dest_params = dest_block->getArguments();

  if (args.size() != dest_params.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.br: Argument count mismatch (passed " 
                 << args.size() << ", target expects " << dest_params.size() << ")\n";
    }
    return false;
  }

  // Copy argument values to target block parameters in the value map
  for (size_t i = 0; i < args.size(); ++i) {
    Value dest_param = dest_params[i];
    Value src_arg = args[i];
    
    if (!value_map.count(src_arg)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ neura.br: Argument " << i 
                     << " (source value) not found in value map\n";
      }
      return false;
    }
    
    // Transfer argument value to target parameter
    value_map[dest_param] = value_map[src_arg];
    if (isVerboseMode() && i < dest_params.size() - 1) {
      llvm::outs() << "[neura-interpreter]  │  ├─ Param[" << i << "]: value = " 
                   << value_map[src_arg].value << "\n";
    } else if (isVerboseMode() && i == dest_params.size() - 1) {
      llvm::outs() << "[neura-interpreter]  │  └─ Param[" << i << "]: value = " 
                   << value_map[src_arg].value << "\n";
    }
  }

  // Update control flow state: last visited = previous current block; current = target block
  last_visited_block = current_block;
  current_block = dest_block;
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Control Transfer\n";
    llvm::outs() << "[neura-interpreter]     └─ Jump successfully to Block@ " << dest_block << "\n";
  }
  return true;
}

/**
 * @brief Handles the execution of a Neura conditional branch operation (neura.cond_br) by transferring control to one of two target blocks based on a boolean condition.
 * 
 * This function processes Neura's conditional branch operations, which direct control flow to a "true" target block or "false" target block based on the value of a boolean condition. 
 * It validates the operation's operands (1 mandatory condition + 1 optional predicate), checks that the condition is a boolean (i1 type), and computes a final predicate to determine if the branch is valid. 
 * If valid, it selects the target block based on the condition's value, passes arguments to the target block's parameters, and updates the current and last visited blocks to reflect the control transfer.
 * 
 * @param op                  The neura.cond_br operation to handle
 * @param value_map           Reference to the map storing values (including the condition and optional predicate)
 * @param current_block       Reference to the current block; updated to the target block after branch
 * @param last_visited_block  Reference to the last visited block; updated to the previous current block after branch
 * @return bool               True if the branch is successfully executed; false for invalid operands, missing values, type mismatches, or invalid predicates
 */
bool handleCondBrOp(neura::CondBr op, llvm::DenseMap<Value, PredicatedData> &value_map, 
                    Block *&current_block, Block *&last_visited_block) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.cond_br:\n";
  }

  if (op.getNumOperands() < 1 || op.getNumOperands() > 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br expects 1 or 2 operands (condition + optional predicate)\n";
    }
    return false;
  }

  auto cond_value = op.getCondition();
  if (!value_map.count(cond_value)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ cond_br: condition value not found in value_map! (SSA name missing)\n";
    }
    return false;
  }
  auto cond_data = value_map[op.getCondition()];

  if (!op.getCondition().getType().isInteger(1)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: condition must be of type i1 (boolean)\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Condition     : value = " << cond_data.value 
               << ", [pred = " << cond_data.predicate << "]\n";
  }

  // Compute final predicate (combines condition's predicate and optional predicate operand)
  bool final_predicate = cond_data.predicate;
  if (op.getNumOperands() > 1) {
    auto pred_data = value_map[op.getPredicate()];
    final_predicate = final_predicate && pred_data.predicate && (pred_data.value != 0.0f);
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Execution Context\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Pred : value = " << pred_data.value
                   << " [pred = " << pred_data.predicate << "]\n";
    }
  }

  // Retrieve successor blocks (targets of the conditional branch)
  auto current_succs_range = current_block->getSuccessors();
  std::vector<Block *> succ_blocks(current_succs_range.begin(), current_succs_range.end());

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Block Information\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Current block : Block@" << current_block << "\n";
    llvm::outs() << "[neura-interpreter]  ├─ Branch Targets\n";
    for (unsigned int i = 0; i < succ_blocks.size(); ++i) {
    if(i < succ_blocks.size() - 1)
      llvm::outs() << "[neura-interpreter]  │  ├─ True block    : Block@" << succ_blocks[i] << "\n";
    else
      llvm::outs() << "[neura-interpreter]  │  └─ False block   : Block@" << succ_blocks[i] << "\n";
    }
  }

  if (!final_predicate) {
    llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: condition or predicate is invalid\n";
    return false;
  }

  // Determine target block based on condition value (non-zero = true branch)
  bool is_true_branch = (cond_data.value != 0.0f);
  Block *target_block = is_true_branch ? op.getTrueDest() : op.getFalseDest();
  const auto &branch_args = is_true_branch ? op.getTrueArgs() : op.getFalseArgs();
  const auto &target_params = target_block->getArguments();
  
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Evaluation\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Condition is " << (cond_data.value != 0.0f ? "true" : "false")
                 << " → selecting '" << (is_true_branch ? "true" : "false") << "' branch\n";
  }
  

  if (branch_args.size() != target_params.size()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.cond_br: argument count mismatch for " 
                   << (is_true_branch ? "true" : "false") << " branch (expected " 
                   << target_params.size() << ", got " << branch_args.size() << ")\n";
    }
    return false;
  }

  // Pass branch arguments to target block parameters (update value_map)
  if (!branch_args.empty()) {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Pass Arguments\n";
    }
  }

  for (size_t i = 0; i < branch_args.size(); ++i) {
    value_map[target_params[i]] = value_map[branch_args[i]];
    if (i < branch_args.size() - 1) {
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  ├─ param[" << i << "]: value = " 
                     << value_map[branch_args[i]].value << "\n";
      }
    } else {
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  │  └─ param[" << i << "]: value = " 
                     << value_map[branch_args[i]].value << "\n";
      }
    }
  }

  // Update control flow state: last visited block = previous current block; current block = target
  last_visited_block = current_block;
  current_block = target_block;
  
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Control Transfer\n";
    llvm::outs() << "[neura-interpreter]     └─ Jump successfully to Block@" << target_block << "\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura phi operation (neura.phi) in control flow mode, selecting the correct input based on the execution path.
 * 
 * This function processes Neura's phi operations, which act as control flow merge points by selecting one of several input values based on the predecessor block
 * that was most recently visited. It identifies the predecessor block that matches the last visited block, retrieves the corresponding input value, and assigns
 * it to the phi operation's result. Validations ensure the current block has predecessors, the last visited block is a valid predecessor, and input count matches
 * predecessor count.
 * 
 * @param op                  The neura.phi operation to handle
 * @param value_map           Reference to the map storing input values and where the phi result will be stored
 * @param current_block       The block containing the phi operation (current execution block)
 * @param last_visited_block  The most recently visited predecessor block (determines which input to select)
 * @return bool               True if the phi operation is successfully processed; false for validation errors (e.g., missing predecessors, mismatched inputs)
 */
bool handlePhiOpControlFlowMode(neura::PhiOp op, llvm::DenseMap<Value, PredicatedData> &value_map, 
                 Block *current_block, Block *last_visited_block) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.phi:\n";
  }

  // Get all predecessor blocks of the current block (possible execution paths leading to this phi)
  auto predecessors_range = current_block->getPredecessors();
  std::vector<Block*> predecessors(predecessors_range.begin(), predecessors_range.end());
  size_t pred_count = predecessors.size(); 

  // Validate current block has at least one predecessor (phi requires merge of paths)
  if (pred_count == 0) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Current block has no predecessors\n";
    }
    return false;
  }

  // Find the index of the last visited block among the predecessors (determines which input to use)
  size_t pred_index = 0;
  bool found = false;
  for (auto pred : predecessors) {
    if (pred == last_visited_block) {
      found = true;
      break;
    }
    ++pred_index;
  }

  // Validate the last visited block is a valid predecessor of the current block
  if (!found) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Last visited block not found in predecessors\n";
    }
    return false;
  }

  auto inputs = op.getInputs();
  size_t input_count = inputs.size();

  // Validate input count matches predecessor count (one input per path)
  if (input_count != pred_count) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Input count (" << input_count 
                   << ") != predecessor count (" << pred_count << ")\n";
    }
    return false;
  }

  // Validate the predecessor index is within the valid range of inputs
  if (pred_index >= input_count) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Invalid predecessor index (" << pred_index << ")\n";
    }
    return false;
  }

  Value input_val = inputs[pred_index];
  if (!value_map.count(input_val)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.phi: Input value not found in value map\n";
    }
    return false;
  }

  // Assign the selected input's data to the phi operation's result
  PredicatedData input_data = value_map[input_val];
  value_map[op.getResult()] = input_data;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Predecessor blocks (" << pred_count << ")\n";
    for (size_t i = 0; i < pred_count; ++i) {
      if(i < pred_count - 1) {
        llvm::outs() << "[neura-interpreter]  │  ├─ [" << i << "]: " << "Block@" << predecessors[i];
      } else {
        llvm::outs() << "[neura-interpreter]  │  └─ [" << i << "]: " << "Block@" << predecessors[i];
      }
      if (i == pred_index) {
        llvm::outs() << " (→ current path)\n";
      } else {
        llvm::outs() << "\n";
      }
    }
    llvm::outs() << "[neura-interpreter]  └─ Result    : " << op.getResult() << "\n";
    llvm::outs() << "[neura-interpreter]     └─ Value : " << input_data.value 
                 << ", [Pred = " << input_data.predicate << "]\n";
  }
  return true;
}

/**
 * @brief Handles the execution of a Neura phi operation (neura.phi) in dataflow mode, merging input values based on their validity.
 * 
 * This function processes Neura's phi operations under dataflow analysis, focusing on merging values from multiple input paths. 
 * It selects the first valid input (with a true predicate) as the result. If no valid inputs exist, it falls back to the first input 
 * but marks its predicate as false. Additionally, it tracks whether the result has changed from its previous state (via `is_updated`) 
 * to support dataflow propagation of changes.
 * 
 * Key dataflow-specific behaviors:
 * - Prioritizes inputs with valid predicates (true) when merging paths.
 * - Explicitly tracks updates to the result to trigger reprocessing of dependent operations.
 * - Gracefully handles missing or invalid inputs by falling back to the first input with an invalid predicate.
 * 
 * @param op          The neura.phi operation to handle
 * @param value_map   Reference to the map storing input values and where the merged result will be stored
 * @return bool       Always returns true, even if no valid inputs are found (handles partial success gracefully)
 */
bool handlePhiOpDataFlowMode(neura::PhiOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.phi(dataflow):\n";
  }

  auto inputs = op.getInputs();
  size_t input_count = inputs.size();

  if (input_count == 0) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: No inputs provided (execution failed)\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Input values (" << input_count << ")\n";
    for (size_t i = 0; i < input_count; ++i) {
      Value input = inputs[i];
      if (value_map.count(input)) {
        auto input_data = value_map[input];
        const std::string prefix = (i < input_count - 1) ? "│ ├─" : "│ └─";
        llvm::outs() << "[neura-interpreter]  " << prefix << "[" << i << "]:"
                     << "value = " << input_data.value << ","
                     << "pred = " << input_data.predicate << "\n";
      } else {
        const std::string prefix = (i < input_count - 1) ? "│ ├─" : "│ └─";
        llvm::outs() << "[neura-interpreter]  " << prefix << "[" << i << "]: <undefined>\n";
      }
    }
  }

  // Initialize result with default values
  PredicatedData result;
  result.value = 0.0f;
  result.predicate = false;
  result.is_vector = false;
  result.vector_data = {};
  result.is_reserve = false;
  result.is_updated = false;
  bool found_valid_input = false;

  // Select the first valid input (with true predicate) to use as the result
  for (size_t i = 0; i < input_count; ++i) { 
    Value input = inputs[i];

    if (!value_map.count(input)) {
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  ├─ Input [" << i << "] not found, skipping\n";
      }
      continue;
    }

    auto input_data = value_map[input];

    // Use the first valid input and stop searching
    if (input_data.predicate && !found_valid_input) {
      result.value = input_data.value;
      result.predicate = input_data.predicate;
      result.is_vector = input_data.is_vector;
      result.vector_data = input_data.vector_data;
      found_valid_input = true;

      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  ├─ Selected input [" << i << "] (latest valid)\n";
      }
      break;
    }
  }

  // Fallback: if no valid inputs, use the first input but mark predicate as false
  if (!found_valid_input && input_count > 0) {
    Value first_input = inputs[0];
    if (value_map.count(first_input)) {
      auto first_data = value_map[first_input];
      result.value = first_data.value;
      result.is_vector = first_data.is_vector;
      result.vector_data = first_data.vector_data;
      result.predicate = false;
      if (isVerboseMode()) {
        llvm::outs() << "[neura-interpreter]  ├─ No valid input, using first input with pred=false\n";
      }
    }
  }

  // Check if the result has changed from its previous state (for dataflow propagation)
  bool fields_changed = false;
  if (value_map.count(op.getResult())) {
    auto old_result = value_map[op.getResult()];
    fields_changed = (result.value != old_result.value) ||
                    (result.predicate != old_result.predicate) ||
                    (result.is_vector != old_result.is_vector) ||
                    (result.vector_data != old_result.vector_data);
  } else {
    fields_changed = true;
  }

  result.is_updated = fields_changed;
  value_map[op.getResult()] = result;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Execution " << (found_valid_input ? "succeeded" : "partially succeeded")
                 << " | Result: value = " << result.value
                 << ", pred = " << result.predicate
                 << ", is_updated = " << result.is_updated << "\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura reservation operation (neura.reserve) by creating a placeholder value.
 * 
 * This function processes Neura's reserve operations, which create a placeholder value with predefined initial properties.
 * The placeholder is initialized with a value of 0.0f, a predicate of false (initially invalid), and is marked as a reserved
 * value via the is_reserve flag. This operation is typically used to allocate or reserve a value slot for future use,
 * where the actual value and validity will be set later.
 * 
 * @param op          The neura.reserve operation to handle
 * @param value_map   Reference to the map where the reserved placeholder will be stored, keyed by the operation's result value
 * @return bool       Always returns true as the reservation operation is guaranteed to succeed
 */
bool handleReserveOp(neura::ReserveOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.reserve:\n";
  }

  PredicatedData placeholder;
  placeholder.value = 0.0f;         /* Initial value set to 0.0f */
  placeholder.predicate = false;    /* Initially marked as invalid (predicate false) */
  placeholder.is_reserve = true;    /* Flag to indicate this is a reserved placeholder */

  Value result = op.getResult();
  value_map[result] = placeholder;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Created placeholder  : " << result << "\n";
    llvm::outs() << "[neura-interpreter]     ├─ Initial value     : 0.0f\n";
    llvm::outs() << "[neura-interpreter]     ├─ Initial predicate : false\n";
    llvm::outs() << "[neura-interpreter]     └─ Type              : " << result.getType() << "\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura control move operation (neura.ctrl_mov) by copying a source value into a reserved target placeholder.
 * 
 * This function processes Neura's ctrl_mov operations, which transfer data from a source value to a target placeholder that was previously reserved via a neura.reserve operation.
 * It validates that the source exists in the value map, the target is a reserved placeholder, and both have matching types. If valid, it copies the source's value, predicate, vector flag,
 * and vector data (if applicable) into the target, updating the reserved placeholder with the source's data.
 * 
 * @param op          The neura.ctrl_mov operation to handle
 * @param value_map   Reference to the map storing both the source value and the target reserved placeholder
 * @return bool       True if the data is successfully moved to the target; false if validation fails (e.g., missing source, invalid target, type mismatch)
 */
bool handleCtrlMovOp(neura::CtrlMovOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.ctrl_mov:\n";
  }

  Value source = op.getValue();
  Value target = op.getTarget();

  if (!value_map.count(source)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.ctrl_mov: Source value not found in value map\n";
    }
    return false;
  }

  if (!value_map.count(target) || !value_map[target].is_reserve) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.ctrl_mov: Target is not a reserve placeholder\n";
    }
    return false;
  }

  const auto &source_data = value_map[source];
  auto &target_data = value_map[target];

  if (source.getType() != target.getType()) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.ctrl_mov: Type mismatch (source ="
                   << source.getType() << ", target =" << target.getType() << ")\n";
    }
    return false;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Source: " << source <<"\n";
    llvm::outs() << "[neura-interpreter]  │  └─ value = "  << source_data.value 
                 << ", [pred = " << source_data.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  ├─ Target: " << target << "\n";
    llvm::outs() << "[neura-interpreter]  │  └─ value = "  << target_data.value 
                 << ", [pred = " << target_data.predicate << "]\n";
  }
  // Copy data from source to target: value, predicate, vector flag, and vector data (if vector)
  target_data.value = source_data.value;
  target_data.predicate = source_data.predicate;
  target_data.is_vector = source_data.is_vector;
  if (source_data.is_vector) {
    target_data.vector_data = source_data.vector_data;
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Updated target placeholder\n";
    llvm::outs() << "[neura-interpreter]     └─ value = "  << target_data.value 
                 << ", [pred = " << target_data.predicate << "]\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura control move operation (neura.ctrl_mov) in dataflow mode, with explicit tracking of value updates.
 * 
 * This function processes neura.ctrl_mov operations under dataflow analysis, focusing on conditional updates to a reserved target placeholder.
 * It only updates the target if the source value's predicate is valid (true). Additionally, it tracks whether the target was actually modified
 * (via the `is_updated` flag) to support dataflow propagation (e.g., triggering reprocessing of dependent operations).
 * 
 * Key differences from standard mode:
 * - Explicitly checks if the source predicate is valid before updating.
 * - Tracks detailed changes (value, predicate, vector status) to set `is_updated`.
 * - Focuses on propagating state changes for dataflow analysis.
 * 
 * @param op          The neura.ctrl_mov operation to handle
 * @param value_map   Reference to the map storing the source value and target reserved placeholder
 * @return bool       True if the operation is processed successfully (including skipped updates); false for critical errors (e.g., missing source/target)
 */
bool handleCtrlMovOpDataFlowMode(neura::CtrlMovOp op, 
                                 llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.ctrl_mov(dataflow):\n";
  }

  Value source = op.getValue();
  if (!value_map.count(source)) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Source value not found (execution failed)\n";
    }
    return false;
  }

  Value target = op.getTarget();
  auto &target_data = value_map[target];
  if (!value_map.count(target) || !target_data.is_reserve) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Target is not a reserve placeholder (execution failed)\n";
    }
    return false; 
  }

  // Capture source data and target's current state (for update checks)
  const auto &source_data = value_map[source];
  const float old_value = target_data.value;
  const bool old_predicate = target_data.predicate;
  const bool old_is_vector = target_data.is_vector;
  const std::vector<float> oldvector_data = target_data.vector_data;

  // Reset update flag; will be set to true only if actual changes occur
  target_data.is_updated = false;
  // Determine if update should proceed: only if source predicate is valid (true)
  const bool should_update = (source_data.predicate == 1);

  if (should_update) {
    // Copy source data to target
    target_data.value = source_data.value;
    target_data.predicate = source_data.predicate;
    target_data.is_vector = source_data.is_vector;
    target_data.vector_data = source_data.is_vector ? source_data.vector_data : std::vector<float>();

    // Check if scalar target was updated (value, predicate, or vector state changed)
    const bool is_scalar_updated = !target_data.is_vector && 
                               (target_data.value != old_value || 
                                target_data.predicate != old_predicate || 
                                !oldvector_data.empty());
    // Check if vector target was updated (predicate or vector elements changed)
    const bool is_vector_updated = target_data.is_vector && 
                               (target_data.predicate != old_predicate || 
                                target_data.vector_data != oldvector_data);
    target_data.is_updated = is_scalar_updated || is_vector_updated || (target_data.is_vector != old_is_vector);
  } else if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Skip update: Source predicate invalid (pred=" 
                 << source_data.predicate << ")\n";
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Source: " << source_data.value << " | " << source_data.predicate << "\n"
                 << "[neura-interpreter]  ├─ Target (after): " << target_data.value << " | " << target_data.predicate 
                 << " | is_updated=" << target_data.is_updated << "\n"
                 << "[neura-interpreter]  └─ Execution succeeded\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura return operation (neura.return) by processing and outputting return values.
 * 
 * This function processes Neura's return operations, which return zero or more values from a function. It retrieves each return value from the value map,
 * validates their existence, and prints them in a human-readable format (scalar or vector) in verbose mode. For vector values, it formats elements as a 
 * comma-separated list, using 0.0f for elements with an invalid predicate. If no return values are present, it indicates a void return.
 * 
 * @param op          The neura.return operation to handle
 * @param value_map   Reference to the map storing the return values to be processed
 * @return bool       True if return values are successfully processed; false if any return value is missing from the value map
 */
bool handleNeuraReturnOp(neura::ReturnOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.return:\n";
  }

  auto return_values = op.getValues();
  // Handle void return (no values)
  if (return_values.empty()) {
    llvm::outs() << "[neura-interpreter]  → Output: (void)\n";
    return true;
  }

  // Collect and validate return values from the value map
  std::vector<PredicatedData> results;
  for (Value val : return_values) {
    if (!value_map.count(val)) {
      llvm::errs() << "[neura-interpreter]  └─ Return value not found in value map\n";
      return false;
    }
    results.push_back(value_map[val]);
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Return values:\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto &data = results[i];
      // Print vector values with predicate check (0.0f if predicate is false)
      if (data.is_vector) {
        llvm::outs() << "[neura-interpreter]  │  └─ vector = [";
        for (size_t j = 0; j < data.vector_data.size(); ++j) {
          float val = data.predicate ? data.vector_data[j] : 0.0f;
          llvm::outs() << llvm::format("%.6f", val);
          if (j != data.vector_data.size() - 1) 
            llvm::outs() << ", ";
        }
        llvm::outs() << "]";
      } else {
        // Print scalar value with predicate check (0.0f if predicate is false)
        float val = data.predicate ? data.value : 0.0f;
        llvm::outs() << "[neura-interpreter]  │  └─" << llvm::format("%.6f", val);
      }
      llvm::outs() << ", [pred = " << data.predicate << "]\n";
    }
    llvm::outs() << "[neura-interpreter]  └─ Execution terminated successfully\n";
  }

  return true;
}

/**
 * @brief Handles the execution of a Neura conditional grant operation (neura.grant_predicate) by updating a value's predicate based on a new condition.
 * 
 * This function processes Neura's grant_predicate operations, which take exactly 2 operands: a source value and a new predicate value. It updates the source value's 
 * predicate to be the logical AND of the source's original predicate, the new predicate's validity, and the new predicate's non-zero value (treating non-zero as true). 
 * The result retains the source's value but uses the computed combined predicate. Errors are returned for invalid operand counts or missing operands in the value map.
 * 
 * @param op          The neura.grant_predicate operation to handle
 * @param value_map   Reference to the map where the updated result will be stored, keyed by the operation's result value
 * @return bool       True if the operation is successfully executed; false for invalid operands or missing entries in the value map
 */
bool handleGrantPredicateOp(neura::GrantPredicateOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.grant_predicate:\n";
  }

  if (op.getOperation()->getNumOperands() != 2) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ neura.grant_predicate expects exactly 2 operands (value, new_predicate)\n";
    } 
    return false;
  }

  if (!value_map.count(op.getValue()) || !value_map.count(op.getPredicate())) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ Error: Source or new predicate not found in value_map\n";
    }
    return false;
  }

  auto source = value_map[op.getValue()];
  auto new_pred = value_map[op.getPredicate()];

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  ├─ Source: value = " << source.value 
                 << ", [pred = " << source.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  │  └─ New predicate: value = " << new_pred.value 
                 << ", [pred = " << new_pred.predicate << "]\n";
  }
  // Evaluate new predicate (non-zero value = true) and compute combined result predicate
  bool is_new_pred_true = (new_pred.value != 0.0f); 
  bool result_predicate = source.predicate && new_pred.predicate && is_new_pred_true;

  PredicatedData result = source;
  result.predicate = result_predicate;
  result.is_vector = source.is_vector; 

  if (isVerboseMode()) {
    std::string grant_status = result_predicate ? "Granted access" : "Denied access (predicate false)";
    llvm::outs() << "[neura-interpreter]  ├─ " << grant_status << "\n";
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value 
                 << ", [pred = " << result_predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura one-time grant operation (neura.grant_once) by granting validity to a value exactly once.
 * 
 * This function processes Neura's grant_once operations, which take either a source value operand or a constant value attribute (but not both). 
 * It grants validity (sets predicate to true) on the first execution and denies validity (sets predicate to false) on all subsequent executions. 
 * The result retains the source/constant value but uses the one-time predicate. Errors are returned for invalid operand/attribute combinations 
 * or unsupported constant types.
 * 
 * @param op          The neura.grant_once operation to handle
 * @param value_map   Reference to the map where the result will be stored, keyed by the operation's result value
 * @return bool       True if the operation is successfully executed; false for invalid inputs or unsupported types
 */
bool handleGrantOnceOp(neura::GrantOnceOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if(isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Executing neura.grant_once:\n";
  }
  // Check if either a value operand or constant attribute is provided
  bool has_value = op.getValue() != nullptr;
  bool has_constant = op.getConstantValue().has_value();
  
  if (has_value == has_constant) {
    if (isVerboseMode()) {
      llvm::errs() << "[neura-interpreter]  └─ grant_once requires exactly one of (value operand, constant_value attribute)\n";
    }
    return false;
  }

  PredicatedData source;
  if (has_value) {
    Value input_value = op.getValue();
    if (!value_map.count(input_value)) {
      if (isVerboseMode()) {
        llvm::errs() << "[neura-interpreter]  └─ Source value not found in value_map\n";
      }
      return false;
    }
    source = value_map[input_value];
    
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
      llvm::outs() << "[neura-interpreter]  │  └─ Source value: " << source.value << ", [pred = " << source.predicate << "]\n";
    }
  } else {
    // Extract and convert constant value from attribute
    Attribute constant_attr = op.getConstantValue().value();
    if (auto int_attr = mlir::dyn_cast<IntegerAttr>(constant_attr)) {
      source.value = int_attr.getInt(); 
      source.predicate = false; 
      source.is_vector = false;

    } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(constant_attr)) {
      source.value = float_attr.getValueAsDouble();
      source.predicate = false;
      source.is_vector = false;
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
  // Track if this operation has already granted access (static to persist across calls)
  static llvm::DenseMap<Operation*, bool> granted;
  bool has_granted = granted[op.getOperation()];
  bool result_predicate = !has_granted;

  if (!has_granted) {
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
  result.predicate = result_predicate;
  result.is_vector = source.is_vector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value 
                 << ", [pred = " << result_predicate << "]\n";
  }

  value_map[op.getResult()] = result;
  return true;
}

/**
 * @brief Handles the execution of a Neura unconditional grant operation (neura.grant_always) by unconditionally validating a value's predicate.
 * 
 * This function processes Neura's grant_always operations, which take exactly 1 operand (a source value) and return a copy of that value with its predicate set to true,
 * regardless of the original predicate. This operation effectively "grants" validity to the value unconditionally. Errors are returned if the operand count is not exactly 1.
 * 
 * @param op          The neura.grant_always operation to handle
 * @param value_map   Reference to the map where the granted result will be stored, keyed by the operation's result value
 * @return bool       True if the operation is successfully executed; false if the operand count is invalid
 */
bool handleGrantAlwaysOp(neura::GrantAlwaysOp op, llvm::DenseMap<Value, PredicatedData> &value_map) {
  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Executing neura.grant_always:\n";
  }

  if (op.getOperation()->getNumOperands() != 1) {
    if (isVerboseMode()) {  
      llvm::errs() << "[neura-interpreter]  └─ neura.grant_always expects exactly 1 operand (value)\n";
    }
    return false;
  }

  auto source = value_map[op.getValue()];
  // Unconditionally set the result predicate to true
  bool result_predicate = true;
  PredicatedData result = source;
  result.predicate = result_predicate;
  result.is_vector = source.is_vector;

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  ├─ Operand\n";
    llvm::outs() << "[neura-interpreter]  │  └─ Source value: " << source.value << ", [pred = " << source.predicate << "]\n";
    llvm::outs() << "[neura-interpreter]  ├─ Granting predicate unconditionally\n";
    llvm::outs() << "[neura-interpreter]  └─ Result: value = " << result.value 
               << ", [pred = " << result_predicate << "]\n";
  }
  

  value_map[op.getResult()] = result;
  return true;
}

bool isDataUpdated(const PredicatedData& old_data, const PredicatedData& new_data) {
  if (old_data.value != new_data.value) return true;
  if (old_data.predicate != new_data.predicate) return true;
  if (old_data.is_vector != new_data.is_vector) return true;
  if (old_data.is_vector && old_data.vector_data != new_data.vector_data) return true;
  return false;
}

/**
 * @brief Executes a single operation, handles type-specific processing, tracks value updates, and propagates changes to dependent operations.
 * 
 * This function processes the given operation using specialized handlers based on its type. It first saves the current values of the operation's
 * results to detect updates after execution. If execution succeeds, it checks for changes in the results' values. If updates are detected,
 * it marks the values as updated and adds all dependent operations (users) to the next work list for subsequent processing.
 * 
 * @param op               The operation to execute
 * @param value_map        Reference to the map storing predicated data for values (tracks current state and updates)
 * @param mem              Reference to the memory object for handling load/store operations
 * @param next_work_list   Reference to the list of operations to process in the next iteration (receives dependent operations on updates)
 * @return bool            True if the operation executes successfully; false if execution fails (e.g., unhandled operation type)
 */
bool executeOperation(Operation* op, 
                     llvm::DenseMap<Value, PredicatedData>& value_map, 
                     Memory& mem,
                     llvm::SmallVector<Operation*, 16>& next_work_list) {
  llvm::DenseMap<Value, PredicatedData> old_values;
  for (Value result : op->getResults()) {
    if (value_map.count(result)) {
      old_values[result] = value_map[result];
    }
  }

  bool execution_success = true;
  if (auto const_op = dyn_cast<mlir::arith::ConstantOp>(op)) {
    execution_success = handleArithConstantOp(const_op, value_map);
  } else if (auto const_op = dyn_cast<neura::ConstantOp>(op)) {
    execution_success = handleNeuraConstantOp(const_op, value_map);
  } else if (auto mov_op = dyn_cast<neura::DataMovOp>(op)) {
    value_map[mov_op.getResult()] = value_map[mov_op.getOperand()];
  } else if (auto add_op = dyn_cast<neura::AddOp>(op)) {
    execution_success = handleAddOp(add_op, value_map);
  } else if (auto sub_op = dyn_cast<neura::SubOp>(op)) {
    execution_success = handleSubOp(sub_op, value_map);
  } else if (auto fadd_op = dyn_cast<neura::FAddOp>(op)) {
    execution_success = handleFAddOp(fadd_op, value_map);
  } else if (auto fsub_op = dyn_cast<neura::FSubOp>(op)) {
    execution_success = handleFSubOp(fsub_op, value_map);
  } else if (auto fmul_op = dyn_cast<neura::FMulOp>(op)) {
    execution_success = handleFMulOp(fmul_op, value_map);
  } else if (auto fdiv_op = dyn_cast<neura::FDivOp>(op)) {
    execution_success = handleFDivOp(fdiv_op, value_map);
  } else if (auto vfmul_op = dyn_cast<neura::VFMulOp>(op)) {
    execution_success = handleVFMulOp(vfmul_op, value_map);
  } else if (auto fadd_fadd_op = dyn_cast<neura::FAddFAddOp>(op)) {
    execution_success = handleFAddFAddOp(fadd_fadd_op, value_map);
  } else if (auto fmul_fadd_op = dyn_cast<neura::FMulFAddOp>(op)) {
    execution_success = handleFMulFAddOp(fmul_fadd_op, value_map);
  } else if (auto ret_op = dyn_cast<func::ReturnOp>(op)) {
    execution_success = handleFuncReturnOp(ret_op, value_map);
  } else if (auto fcmp_op = dyn_cast<neura::FCmpOp>(op)) {
    execution_success = handleFCmpOp(fcmp_op, value_map);
  } else if (auto icmp_op = dyn_cast<neura::ICmpOp>(op)) {
    execution_success = handleICmpOp(icmp_op, value_map);
  } else if (auto or_op = dyn_cast<neura::OrOp>(op)) {
    execution_success = handleOrOp(or_op, value_map);
  } else if (auto not_op = dyn_cast<neura::NotOp>(op)) {
    execution_success = handleNotOp(not_op, value_map);
  } else if (auto sel_op = dyn_cast<neura::SelOp>(op)) {
    execution_success = handleSelOp(sel_op, value_map);
  } else if (auto cast_op = dyn_cast<neura::CastOp>(op)) {
    execution_success = handleCastOp(cast_op, value_map);
  } else if (auto load_op = dyn_cast<neura::LoadOp>(op)) {
    execution_success = handleLoadOp(load_op, value_map, mem);
  } else if (auto store_op = dyn_cast<neura::StoreOp>(op)) {
    execution_success = handleStoreOp(store_op, value_map, mem);
  } else if (auto gep_op = dyn_cast<neura::GEP>(op)) {
    execution_success = handleGEPOp(gep_op, value_map);
  } else if (auto load_index_op = dyn_cast<neura::LoadIndexedOp>(op)) {
    execution_success = handleLoadIndexedOp(load_index_op, value_map, mem);
  } else if (auto store_index_op = dyn_cast<neura::StoreIndexedOp>(op)) {
    execution_success = handleStoreIndexedOp(store_index_op, value_map, mem);
  } else if (auto br_op = dyn_cast<neura::Br>(op)) {
    // execution_success = handleBrOp(br_op, value_map, current_block, last_visited_block);
  } else if (auto cond_br_op = dyn_cast<neura::CondBr>(op)) {
    // execution_success = handleCondBrOp(cond_br_op, value_map, current_block, last_visited_block);
  } else if (auto phi_op = dyn_cast<neura::PhiOp>(op)) {
    execution_success = handlePhiOpDataFlowMode(phi_op, value_map);
  } else if (auto reserve_op = dyn_cast<neura::ReserveOp>(op)) {
    execution_success = handleReserveOp(reserve_op, value_map);
  } else if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(op)) {
    execution_success = handleCtrlMovOpDataFlowMode(ctrl_mov_op, value_map);
  } else if (auto return_op = dyn_cast<neura::ReturnOp>(op)) {
    execution_success = handleNeuraReturnOp(return_op, value_map);
  } else if (auto grant_pred_op = dyn_cast<neura::GrantPredicateOp>(op)) {
    execution_success = handleGrantPredicateOp(grant_pred_op, value_map);
  } else if (auto grant_once_op = dyn_cast<neura::GrantOnceOp>(op)) {
    execution_success = handleGrantOnceOp(grant_once_op, value_map);
  } else if (auto grant_always_op = dyn_cast<neura::GrantAlwaysOp>(op)) {
    execution_success = handleGrantAlwaysOp(grant_always_op, value_map);
  } else {
    llvm::errs() << "[neura-interpreter]  Unhandled op: ";
    op->print(llvm::errs());
    llvm::errs() << "\n";
    execution_success = false;
  }

  // If execution failed, exit early without propagating updates
  if (!execution_success) {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Operation failed, no propagation\n";
    }
    return execution_success;
  }

  // Check if any of the operation's results were updated during execution
  bool has_update = false;
  for (Value result : op->getResults()) {
    if (!value_map.count(result)) continue;
    PredicatedData new_data = value_map[result];

    // New result (not in old_values) is considered an update
    if (!old_values.count(result)) {
      new_data.is_updated = true;
      value_map[result] = new_data; 
      has_update = true;
      continue;
    }

    // Compare new value with old value to detect updates
    PredicatedData old_data = old_values[result];
    if (isDataUpdated(old_data, new_data)) {
      new_data.is_updated = true;
      value_map[result] = new_data;
      has_update = true;
    } else {
      new_data.is_updated = false;
      value_map[result] = new_data;
    }
  }

  // Special case: check for updates in operations with no results (e.g., CtrlMovOp)
  if (op->getNumResults() == 0) {
    if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(op)) {
      Value target = ctrl_mov_op.getTarget();
      if (value_map.count(target) && value_map[target].is_updated) {
        has_update = true;
      } else {
        llvm::outs() << "[neura-interpreter]  No update for ctrl_mov target: " << target << "\n";
      }
    }
  }

  // If updates occurred, propagate to dependent operations (users)
  if (has_update) {
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Operation updated, propagating to users...\n";
    }

    // Collect all values affected by the update
    llvm::SmallVector<Value, 4> affected_values;
    for (Value result : op->getResults()) {
      if (value_map.count(result) && value_map[result].is_updated) {
        affected_values.push_back(result);
      }
    }

    // Include targets from the operations without result (e.g., CtrlMovOp)
    if (op->getNumResults() == 0) {
      if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(op)) {
        Value target = ctrl_mov_op.getTarget();
        if (value_map.count(target) && value_map[target].is_updated) {
          affected_values.push_back(target);
        }
      }
    }

     // Add all users of affected values to the next work list (if not already present)
    for (Value val : affected_values) {
      if (!value_users.count(val)) continue;
      for (Operation* user_op : value_users[val]) {
        if (!in_work_list[user_op]) {
          next_work_list.push_back(user_op);
          in_work_list[user_op] = true;
          if (isVerboseMode()) {
            llvm::outs() << "[neura-interpreter]  Added user to next work_list: ";
            user_op->print(llvm::outs());
            llvm::outs() << "\n";
          }
        }
      }
    }
  }

  return execution_success;
}

/**
 * @brief Executes a function in data flow mode, processing operations based on data availability and managing return operations separately.
 * 
 * This function processes operations in a work list, where each operation is executed once its dependencies are satisfied. 
 * Return operations are held in a delay queue until all their input values are valid, ensuring they execute only when all prerequisites are met.
 * It iteratively processes the work list, propagating data through the graph until all operations (including returns) are completed.
 * 
 * @param func        The function to execute in data flow mode
 * @param value_map   Reference to a map storing predicated data for each value (tracks valid/invalid state)
 * @param mem         Reference to the memory object for handling load/store operations
 * @return int        0 if execution completes successfully, 1 if an error occurs during operation execution
 */
int runDataFlowMode(func::FuncOp func,
                    llvm::DenseMap<Value, PredicatedData>& value_map,
                    Memory& mem) {
  // Queue to hold return operations until their dependencies are satisfied
  llvm::SmallVector<Operation*, 4> delay_queue;

  // Initialize work list with all operations except return ops (which go to delay queue)
  for (auto& block : func.getBody()) {
    for (auto& op : block.getOperations()) {
      if (isa<neura::ReturnOp>(op) || isa<func::ReturnOp>(op)) {
        delay_queue.push_back(&op);
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  Return op added to delay queue: ";
          op.print(llvm::outs());
          llvm::outs() << "\n";
        }
      } else {
        // Add non-return operations to the work list and mark them as present
        work_list.push_back(&op);
        in_work_list[&op] = true;
      }
    }
  }

  int iter_count = 0;
  // Main loop: process work list and delay queue until both are empty
  while (!work_list.empty() || !delay_queue.empty()) {
    iter_count++;
    if (isVerboseMode()) {
      llvm::outs() << "[neura-interpreter]  Iteration " << iter_count
                   << " | work_list: " << work_list.size()
                   << " | delay_queue: " << delay_queue.size() << "\n";
    }

    llvm::SmallVector<Operation*, 16> next_work_list;

    // Process all operations in the current work list
    for (Operation* op : work_list) {
      in_work_list[op] = false;
      if (!executeOperation(op, value_map, mem, next_work_list)) {
        return 1;
      }
    }

    // Check if return operations in the delay queue can be executed
    if (!delay_queue.empty()) {
      Operation* return_op = delay_queue[0];
      bool can_execute_return = true;
      // Verify all input operands of the return op are valid (exist and have true predicate)
      for (Value input : return_op->getOperands()) {
        if (!value_map.count(input) || !value_map[input].predicate) {
          can_execute_return = false;
          break;
        }
      }

      if (can_execute_return) {
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  All return inputs are valid, executing return\n";
        }
        for (Operation* op : delay_queue) {
          next_work_list.push_back(op);
          in_work_list[op] = true;
        }
        delay_queue.clear();
      } else {
        if (isVerboseMode()) {
          llvm::outs() << "[neura-interpreter]  Some return inputs are invalid, keeping return in delay queue\n";
        }
      }
    }

    // Prepare work list for the next iteration
    work_list = std::move(next_work_list);
  }

  if (isVerboseMode()) {
    llvm::outs() << "[neura-interpreter]  Total iterations: " << iter_count << "\n";
  }

  return 0;
}

/**
 * @brief Executes the control flow of a function by processing operations in sequence and handling control transfers.
 * 
 * This function iterates through the operations within the function's blocks, processing each operation
 * according to its type using specialized handler functions. It manages control flow transfers (branches,
 * conditional branches) and maintains the current execution state including the active block and operation index.
 * 
 * @param func        The function to execute in control flow mode
 * @param value_map   Reference to a map storing predicated data for each value
 * @param mem         Reference to the memory object for handling load/store operations
 * @return int        0 if execution completes successfully, 1 if an error occurs (unhandled operation or handler failure)
 */
int runControlFlowMode(func::FuncOp func, 
                      llvm::DenseMap<Value, PredicatedData>& value_map, 
                      Memory& mem) {
  Block* current_block = &func.getBody().front(); /* Initialize execution state - start at the first block of the function */
  Block* last_visited_block = nullptr;            /* Track the previous block for handling phi operations */
  size_t op_index = 0;                            /* Current position within the operations of the current block */
  bool is_terminated = false;                     /* Flag to indicate if execution has been terminated (e.g., by return operation) */
  
  // Main execution loop - continues until termination or no more blocks to process
  while (!is_terminated && current_block) {
    auto& operations = current_block->getOperations();

    if (op_index >= operations.size()) {
      break;
    }

    Operation& op = *std::next(operations.begin(), op_index);

    // Handle each operation type with specialized handlers
    if (auto const_op = dyn_cast<mlir::arith::ConstantOp>(op)) {
      if (!handleArithConstantOp(const_op, value_map)) return 1;
      ++op_index;
    } else if (auto const_op = dyn_cast<neura::ConstantOp>(op)) {
      if (!handleNeuraConstantOp(const_op, value_map)) return 1;
      ++op_index;
    } else if (auto mov_op = dyn_cast<neura::DataMovOp>(op)) {
      value_map[mov_op.getResult()] = value_map[mov_op.getOperand()];
      ++op_index;
    } else if (auto add_op = dyn_cast<neura::AddOp>(op)) {
      if (!handleAddOp(add_op, value_map)) return 1;
      ++op_index;
    } else if (auto sub_op = dyn_cast<neura::SubOp>(op)) {
      if (!handleSubOp(sub_op, value_map)) return 1;
      ++op_index;
    } else if (auto fadd_op = dyn_cast<neura::FAddOp>(op)) {
      if (!handleFAddOp(fadd_op, value_map)) return 1;
      ++op_index;
    } else if (auto fsub_op = dyn_cast<neura::FSubOp>(op)) {
      if (!handleFSubOp(fsub_op, value_map)) return 1;
      ++op_index;
    } else if (auto fmul_op = dyn_cast<neura::FMulOp>(op)) {
      if (!handleFMulOp(fmul_op, value_map)) return 1;
      ++op_index;
    } else if (auto fdiv_op = dyn_cast<neura::FDivOp>(op)) {
      if (!handleFDivOp(fdiv_op, value_map)) return 1;
      ++op_index;
    } else if (auto vfmul_op = dyn_cast<neura::VFMulOp>(op)) {
      if (!handleVFMulOp(vfmul_op, value_map)) return 1;
      ++op_index;
    } else if (auto fadd_fadd_op = dyn_cast<neura::FAddFAddOp>(op)) {
      if (!handleFAddFAddOp(fadd_fadd_op, value_map)) return 1;
      ++op_index;
    } else if (auto fmul_fadd_op = dyn_cast<neura::FMulFAddOp>(op)) {
      if (!handleFMulFAddOp(fmul_fadd_op, value_map)) return 1;
      ++op_index;
    } else if (auto ret_op = dyn_cast<func::ReturnOp>(op)) {
      if (!handleFuncReturnOp(ret_op, value_map)) return 1;
      is_terminated = true;
      ++op_index;
    } else if (auto fcmp_op = dyn_cast<neura::FCmpOp>(op)) {
      if (!handleFCmpOp(fcmp_op, value_map)) return 1;
      ++op_index;
    } else if (auto icmp_op = dyn_cast<neura::ICmpOp>(op)) {
      if (!handleICmpOp(icmp_op, value_map)) return 1;
      ++op_index;
    } else if (auto or_op = dyn_cast<neura::OrOp>(op)) {
      if (!handleOrOp(or_op, value_map)) return 1;
      ++op_index;
    } else if (auto not_op = dyn_cast<neura::NotOp>(op)) {
      if (!handleNotOp(not_op, value_map)) return 1;
      ++op_index;
    } else if (auto sel_op = dyn_cast<neura::SelOp>(op)) {
      if (!handleSelOp(sel_op, value_map)) return 1;
      ++op_index;
    } else if (auto cast_op = dyn_cast<neura::CastOp>(op)) {
      if (!handleCastOp(cast_op, value_map)) return 1;
      ++op_index;
    } else if (auto load_op = dyn_cast<neura::LoadOp>(op)) {
      if (!handleLoadOp(load_op, value_map, mem)) return 1;
      ++op_index;
    } else if (auto store_op = dyn_cast<neura::StoreOp>(op)) {
      if (!handleStoreOp(store_op, value_map, mem)) return 1;
      ++op_index;
    } else if (auto gep_op = dyn_cast<neura::GEP>(op)) {
      if (!handleGEPOp(gep_op, value_map)) return 1;
      ++op_index;
    } else if (auto load_index_op = dyn_cast<neura::LoadIndexedOp>(op)) {
      if (!handleLoadIndexedOp(load_index_op, value_map, mem)) return 1;
      ++op_index;
    } else if (auto store_index_op = dyn_cast<neura::StoreIndexedOp>(op)) {
      if (!handleStoreIndexedOp(store_index_op, value_map, mem)) return 1;
      ++op_index;
    } else if (auto br_op = dyn_cast<neura::Br>(op)) {
      // Branch operations change the current block - reset index to start of new block
      if (!handleBrOp(br_op, value_map, current_block, last_visited_block)) return 1;
      op_index = 0;  
    } else if (auto cond_br_op = dyn_cast<neura::CondBr>(op)) {
      // Conditional branches change the current block - reset index to start of new block
      if (!handleCondBrOp(cond_br_op, value_map, current_block, last_visited_block)) return 1;
      op_index = 0; 
    } else if (auto phi_op = dyn_cast<neura::PhiOp>(op)) {
      // Phi operations depend on previous block for value selection
      if (!handlePhiOpControlFlowMode(phi_op, value_map, current_block, last_visited_block)) return 1;
      ++op_index;
    } else if (auto reserve_op = dyn_cast<neura::ReserveOp>(op)) {
      if (!handleReserveOp(reserve_op, value_map)) return 1;
      ++op_index;
    } else if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(op)) {
      if (!handleCtrlMovOp(ctrl_mov_op, value_map)) return 1;
      ++op_index;
    } else if (auto return_op = dyn_cast<neura::ReturnOp>(op)) {
      if (!handleNeuraReturnOp(return_op, value_map)) return 1;
      is_terminated = true;
      ++op_index;
    } else if (auto grant_pred_op = dyn_cast<neura::GrantPredicateOp>(op)) {
      if (!handleGrantPredicateOp(grant_pred_op, value_map)) return 1;
      ++op_index;
    } else if (auto grant_once_op = dyn_cast<neura::GrantOnceOp>(op)) {
      if (!handleGrantOnceOp(grant_once_op, value_map)) return 1;
      ++op_index;
    } else if (auto grant_always_op = dyn_cast<neura::GrantAlwaysOp>(op)) {
      if (!handleGrantAlwaysOp(grant_always_op, value_map)) return 1;
      ++op_index;
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

  llvm::SourceMgr source_mgr;
  auto file_or_err = mlir::openInputFile(argv[1]);
  if (!file_or_err) {
    llvm::errs() << "[neura-interpreter]  Error opening file\n";
    return 1;
  }

  source_mgr.AddNewSourceBuffer(std::move(file_or_err), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(source_mgr, &context);
  if (!module) {
    llvm::errs() << "[neura-interpreter]  Failed to parse MLIR input file\n";
    return 1;
  }

  // Changes map to store PredicatedData instead of just float.
  llvm::DenseMap<Value, PredicatedData> value_map;

  Memory mem(1024); // 1MB

  if (isDataflowMode()) {
    buildDependencyGraph(*module);
    for (auto func : module->getOps<func::FuncOp>()) {
      if (runDataFlowMode(func, value_map, mem)) {
        llvm::errs() << "[neura-interpreter]  Data Flow execution failed\n";
        return 1;
      }
    }
  } else {
    for (auto func : module->getOps<func::FuncOp>()) {
      if (runControlFlowMode(func, value_map, mem)) {
        llvm::errs() << "[neura-interpreter]  Control Flow execution failed\n";
        return 1;
      }
    }
  }

  return 0;
}