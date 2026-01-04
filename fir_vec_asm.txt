# Compiled II: 5

PE(0,1):
{
  DATA_MOV, [EAST, RED] -> [$0] (t=7, inv_iters=1)
  DATA_MOV, [NORTH, RED] -> [$1] (t=7, inv_iters=1)
} (idx_per_ii=2)
{
  GRANT_PREDICATE, [$0], [$1] -> [NORTH, RED] (t=8, inv_iters=1)
} (idx_per_ii=3)

PE(1,1):
{
  VADD, [EAST, RED], [NORTH, RED] -> [WEST, RED], [NORTH, RED] (t=6, inv_iters=1)
} (idx_per_ii=1)
{
  RETURN_VALUE, [NORTH, RED] (t=9, inv_iters=1)
} (idx_per_ii=4)

PE(2,1):
{
  DATA_MOV, [EAST, RED] -> [WEST, RED] (t=5, inv_iters=1)
} (idx_per_ii=0)
{
  GEP, [NORTH, RED] -> [$0] (t=2, inv_iters=0)
} (idx_per_ii=2)
{
  LOAD, [$0] -> [EAST, RED] (t=3, inv_iters=0)
} (idx_per_ii=3)

PE(3,1):
{
  VMUL, [NORTH, RED], [WEST, RED] -> [WEST, RED] (t=4, inv_iters=0)
} (idx_per_ii=4)

PE(0,2):
{
  DATA_MOV, [EAST, RED] -> [SOUTH, RED] (t=6, inv_iters=1)
} (idx_per_ii=1)
{
  GRANT_ONCE, [] -> [$0] (t=3, inv_iters=0)
} (idx_per_ii=3)
{
  PHI_START, [$0], [SOUTH, RED] -> [EAST, RED] (t=4, inv_iters=0)
} (idx_per_ii=4)

PE(1,2):
{
  DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=5, inv_iters=1)
  DATA_MOV, [EAST, RED] -> [WEST, RED] (t=5, inv_iters=1)
} (idx_per_ii=0)
{
  GRANT_PREDICATE, [SOUTH, RED], [$0] -> [$0] (t=7, inv_iters=1)
} (idx_per_ii=2)
{
  VECTOR.REDUCE.ADD, [$0] -> [SOUTH, RED] (t=8, inv_iters=1)
} (idx_per_ii=3)
{
  DATA_MOV, [EAST, RED] -> [$0] (t=4, inv_iters=0)
} (idx_per_ii=4)

PE(2,2):
{
  GRANT_PREDICATE, [$1], [$0] -> [$0] (t=5, inv_iters=1)
} (idx_per_ii=0)
{
  PHI_START, [EAST, RED], [$0] -> [SOUTH, RED], [EAST, RED], [$0] (t=1, inv_iters=0)
} (idx_per_ii=1)
{
  ADD, [$0], [#4] -> [$0], [$1] (t=2, inv_iters=0)
} (idx_per_ii=2)
{
  ICMP_EQ, [$0], [#32] -> [$0], [WEST, RED] (t=3, inv_iters=0)
} (idx_per_ii=3)
{
  NOT, [$0] -> [$0], [WEST, RED] (t=4, inv_iters=0)
} (idx_per_ii=4)

PE(3,2):
{
  GRANT_ONCE, [#0] -> [WEST, RED] (t=0, inv_iters=0)
} (idx_per_ii=0)
{
  GEP, [WEST, RED] -> [$0] (t=2, inv_iters=0)
} (idx_per_ii=2)
{
  LOAD, [$0] -> [SOUTH, RED] (t=3, inv_iters=0)
} (idx_per_ii=3)

