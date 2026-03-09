# Compiled II: 11

PE(0,0):
{
  GRANT_ONCE, [#0] -> [EAST, RED], [$0] (t=0, inv_iters=0)
} (idx_per_ii=0)
{
  PHI_START, [$0], [NORTH, RED] -> [$0], [NORTH, RED] (t=1, inv_iters=0)
} (idx_per_ii=1)
{
  SHL, [$0], [#4] -> [$0] (t=2, inv_iters=0)
} (idx_per_ii=2)
{
  GEP, [$0] -> [NORTH, RED] (t=3, inv_iters=0)
} (idx_per_ii=3)
{
  DATA_MOV, [NORTH, RED] -> [$0] (t=7, inv_iters=0)
} (idx_per_ii=7)
{
  GEP, [$0] -> [EAST, RED] (t=9, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=9, inv_iters=0)
} (idx_per_ii=9)

PE(1,0):
{
  DATA_MOV, [WEST, RED] -> [$0] (t=1, inv_iters=0)
  CTRL_MOV, [NORTH, RED] -> [$1] (t=12, inv_iters=1)
} (idx_per_ii=1)
{
  PHI_START, [$0], [$1] -> [$0], [NORTH, RED] (t=2, inv_iters=0)
} (idx_per_ii=2)
{
  DATA_MOV, [NORTH, RED] -> [$1] (t=6, inv_iters=0)
} (idx_per_ii=6)
{
  PHI_START, [$0], [$2] -> [$2], [$0] (t=8, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [WEST, RED] (t=8, inv_iters=0)
  DATA_MOV, [NORTH, RED] -> [$3] (t=8, inv_iters=0)
} (idx_per_ii=8)
{
  GRANT_PREDICATE, [$0], [$1] -> [NORTH, RED] (t=9, inv_iters=0)
} (idx_per_ii=9)
{
  GRANT_PREDICATE, [$2], [$3] -> [$2] (t=10, inv_iters=0)
  DATA_MOV, [WEST, RED] -> [EAST, RED] (t=10, inv_iters=0)
} (idx_per_ii=10)

PE(2,0):
{
  STORE, [NORTH, RED], [WEST, RED] (t=11, inv_iters=1)
} (idx_per_ii=0)
{
  DATA_MOV, [NORTH, RED] -> [WEST, RED] (t=7, inv_iters=0)
} (idx_per_ii=7)

PE(0,1):
{
  GRANT_PREDICATE, [$3], [$0] -> [SOUTH, RED] (t=11, inv_iters=1)
  CTRL_MOV, [EAST, RED] -> [$1] (t=11, inv_iters=1)
} (idx_per_ii=0)
{
  DATA_MOV, [SOUTH, RED] -> [$0] (t=2, inv_iters=0)
} (idx_per_ii=2)
{
  PHI_START, [SOUTH, RED], [$2] -> [EAST, RED], [$2] (t=4, inv_iters=0)
} (idx_per_ii=4)
{
  PHI_START, [$0], [$1] -> [EAST, RED], [$0] (t=5, inv_iters=0)
} (idx_per_ii=5)
{
  GRANT_PREDICATE, [$0], [EAST, RED] -> [SOUTH, RED], [$0] (t=6, inv_iters=0)
} (idx_per_ii=6)
{
  ADD, [$0], [#1] -> [$0], [$3] (t=7, inv_iters=0)
} (idx_per_ii=7)
{
  ICMP_EQ, [$0], [#4] -> [EAST, RED], [$0], [$1] (t=8, inv_iters=0)
} (idx_per_ii=8)
{
  GRANT_PREDICATE, [$0], [$1] -> [NORTH, RED] (t=9, inv_iters=0)
} (idx_per_ii=9)
{
  GRANT_PREDICATE, [$2], [SOUTH, RED] -> [$2] (t=10, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [$0] (t=10, inv_iters=0)
} (idx_per_ii=10)

PE(1,1):
{
  GRANT_PREDICATE, [$1], [$3] -> [SOUTH, RED] (t=11, inv_iters=1)
} (idx_per_ii=0)
{
  PHI_START, [SOUTH, RED], [$0] -> [$1], [EAST, RED], [$0] (t=3, inv_iters=0)
} (idx_per_ii=3)
{
  ADD, [$0], [#1] -> [$0], [EAST, RED] (t=4, inv_iters=0)
} (idx_per_ii=4)
{
  ICMP_EQ, [$0], [#4] -> [EAST, RED], [WEST, RED], [$2], [SOUTH, RED], [NORTH, RED] (t=5, inv_iters=0)
  DATA_MOV, [WEST, RED] -> [$0] (t=5, inv_iters=0)
} (idx_per_ii=5)
{
  GEP, [$0], [$1] -> [$0] (t=6, inv_iters=0)
  DATA_MOV, [WEST, RED] -> [$1] (t=6, inv_iters=0)
  DATA_MOV, [$2] -> [EAST, RED] (t=6, inv_iters=0)
} (idx_per_ii=6)
{
  LOAD, [$0] -> [EAST, RED] (t=7, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [SOUTH, RED] (t=7, inv_iters=0)
} (idx_per_ii=7)
{
  CTRL_MOV, [EAST, RED] -> [$0] (t=8, inv_iters=0)
} (idx_per_ii=8)
{
  NOT, [WEST, RED] -> [WEST, RED], [$3], [NORTH, RED] (t=9, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [$2] (t=9, inv_iters=0)
} (idx_per_ii=9)
{
  GRANT_PREDICATE, [$1], [$2] -> [WEST, RED] (t=10, inv_iters=0)
  DATA_MOV, [SOUTH, RED] -> [$1] (t=10, inv_iters=0)
} (idx_per_ii=10)

PE(2,1):
{
  GRANT_PREDICATE, [$1], [$3] -> [NORTH, RED] (t=11, inv_iters=1)
} (idx_per_ii=0)
{
  DATA_MOV, [WEST, RED] -> [$0] (t=4, inv_iters=0)
} (idx_per_ii=4)
{
  GEP, [$0] -> [EAST, RED] (t=5, inv_iters=0)
  DATA_MOV, [WEST, RED] -> [$0] (t=5, inv_iters=0)
} (idx_per_ii=5)
{
  NOT, [WEST, RED] -> [$1], [$3], [SOUTH, RED], [EAST, RED], [WEST, RED], [NORTH, RED] (t=6, inv_iters=0)
} (idx_per_ii=6)
{
  GRANT_PREDICATE, [$0], [$1] -> [WEST, RED] (t=7, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [$0] (t=7, inv_iters=0)
  DATA_MOV, [WEST, RED] -> [$2] (t=7, inv_iters=0)
} (idx_per_ii=7)
{
  MUL, [$0], [WEST, RED] -> [$0] (t=8, inv_iters=0)
  DATA_MOV, [NORTH, RED] -> [$1] (t=8, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [WEST, RED] (t=8, inv_iters=0)
} (idx_per_ii=8)
{
  ADD, [$0], [$1] -> [$1], [$0] (t=9, inv_iters=0)
} (idx_per_ii=9)
{
  GRANT_PREDICATE, [$0], [$2] -> [SOUTH, RED] (t=10, inv_iters=0)
} (idx_per_ii=10)

PE(3,1):
{
  LOAD, [WEST, RED] -> [WEST, RED] (t=6, inv_iters=0)
} (idx_per_ii=6)
{
  DATA_MOV, [WEST, RED] -> [WEST, RED] (t=7, inv_iters=0)
} (idx_per_ii=7)

PE(0,2):
{
  RETURN_VOID, [SOUTH, RED] (t=10, inv_iters=0)
} (idx_per_ii=10)

PE(1,2):
{
  DATA_MOV, [SOUTH, RED] -> [$0] (t=6, inv_iters=0)
} (idx_per_ii=6)
{
  GRANT_PREDICATE, [EAST, RED], [$0] -> [$0] (t=9, inv_iters=0)
} (idx_per_ii=9)
{
  GRANT_PREDICATE, [$0], [SOUTH, RED] -> [EAST, RED] (t=10, inv_iters=0)
} (idx_per_ii=10)

PE(2,2):
{
  CTRL_MOV, [WEST, RED] -> [$1] (t=11, inv_iters=1)
} (idx_per_ii=0)
{
  CTRL_MOV, [SOUTH, RED] -> [$2] (t=12, inv_iters=1)
} (idx_per_ii=1)
{
  GRANT_ONCE, [#0] -> [$0] (t=5, inv_iters=0)
} (idx_per_ii=5)
{
  PHI_START, [$0], [$1] -> [$1], [$0] (t=6, inv_iters=0)
} (idx_per_ii=6)
{
  PHI_START, [$0], [$2] -> [SOUTH, RED] (t=7, inv_iters=0)
  DATA_MOV, [SOUTH, RED] -> [$2] (t=7, inv_iters=0)
} (idx_per_ii=7)
{
  PHI_START, [$1], [$3] -> [$0], [WEST, RED] (t=8, inv_iters=0)
} (idx_per_ii=8)
{
  GRANT_PREDICATE, [$0], [$2] -> [$3] (t=10, inv_iters=0)
} (idx_per_ii=10)

