# Mapping Optimization Summary

## Performance Improvements Achieved

### Core Algorithm Changes
1. **Degree-Based Priority Scheduling**: Modified `flatten_level_buckets()` to sort operations within each ALAP level by their connectivity degree (fan-in + fan-out), ensuring high-connectivity operations get mapped first.

2. **Link Congestion Awareness**: Added a balanced quadratic penalty in `calculateAward()` to guide the mapper away from congested network areas without being overly restrictive.

3. **Stable Tie-Breaking**: Implemented deterministic tie-breaking in both operation scheduling and award sorting to minimize test instability.

## Performance Results

### Tests with Improved II
| Test | Original II | New II | Improvement |
|------|-------------|--------|-------------|
| `test/neura/fusion/test.mlir` | 13 | 11 | **-15.4%** |
| `test/c2llvm2mlir/nested_loop/test.mlir` | 13 | 11 | **-15.4%** |
| `test/code_gen/test_code_generate.mlir` | 5 | 4 | **-20.0%** |

### Tests Maintaining Optimal II
| Test | II | Status |
|------|-----|--------|
| `test/neura/ctrl/branch_for.mlir` | 4 | Maintained (at RecMII) |
| `test/e2e/relu/relu_kernel.mlir` | 5 | Maintained (at RecMII) |
| `test/e2e/bicg/bicg_kernel.mlir` | 11 | Maintained |
| `test/e2e/fir/fir_kernel.mlir` | 5 | Maintained (at RecMII) |
| `test/e2e/histogram/histogram_kernel.mlir` | 5 | Maintained (at RecMII) |

## Test Status
- **Passed**: 71/87 (81.61%)
- **Updated and Passing**: 1 test (`nested_loop/test.mlir`)
- **Remaining to Update**: ~11 tests with mapping layout changes

## Key Code Modifications

### File: `lib/NeuraDialect/Mapping/mapping_util.cpp`

#### 1. Degree-Based Sorting (lines 395-428)
```cpp
std::vector<std::pair<Operation *, int>> mlir::neura::flatten_level_buckets(
    const std::vector<std::vector<Operation *>> &level_buckets) {
  // ... 
  // Sort by degree (num_operands + num_users) with stable tie-breaking
  std::sort(ops_with_index.begin(), ops_with_index.end(),
            [](const std::pair<Operation *, int> &a_pair,
               const std::pair<Operation *, int> &b_pair) {
              // Higher degree operations first
              if (degree_a != degree_b)
                return degree_a > degree_b;
              return a_pair.second < b_pair.second; // Stable tie-breaker
            });
}
```

#### 2. Link Congestion Penalty (lines 993-1018)
```cpp
// Balanced quadratic penalty based on link occupancy
int congestion_penalty = static_cast<int>(in_ratio * in_ratio * 10) +
                         static_cast<int>(out_ratio * out_ratio * 10);
```

#### 3. Stable Award Sorting (lines 1024-1036)
```cpp
std::sort(locs_award_vec.begin(), locs_award_vec.end(),
          [](const std::pair<MappingLoc, int> &a,
             const std::pair<MappingLoc, int> &b) {
            if (a.second != b.second)
              return a.second > b.second;
            return a.first.time_step < b.first.time_step; // Tie-breaker
          });
```

## Next Steps
1. Continue updating remaining test expectations for layout changes
2. Consider filing issue/PR for the performance improvements
3. Monitor for any regressions in edge cases
