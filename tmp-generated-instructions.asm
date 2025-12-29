# Compiled II: 4

PE(0,0):
{
  CONSTANT, [#0] -> [$0] (t=0, inv_iters=0)
} (idx_per_ii=0)
{
  GRANT_ONCE, [$0] -> [EAST, RED] (t=1, inv_iters=0)
  DATA_MOV, [NORTH, RED] -> [$0] (t=5, inv_iters=1)
} (idx_per_ii=1)
{
  GRANT_PREDICATE, [$0], [NORTH, RED] -> [NORTH, RED] (t=7, inv_iters=1)
} (idx_per_ii=3)

PE(1,0):
{
  DATA_MOV, [NORTH, RED] -> [EAST, RED] (t=4, inv_iters=1)
} (idx_per_ii=0)
{
  PHI_START, [WEST, RED], [EAST, RED] -> [NORTH, RED] (t=2, inv_iters=0)
} (idx_per_ii=2)

PE(2,0):
{
  GRANT_PREDICATE, [WEST, RED], [NORTH, RED] -> [WEST, RED] (t=5, inv_iters=1)
} (idx_per_ii=1)
{
  PHI_START, [EAST, RED], [NORTH, RED] -> [NORTH, RED] (t=3, inv_iters=0)
} (idx_per_ii=3)

PE(3,0):
{
  CONSTANT, [#10] -> [$0] (t=1, inv_iters=0)
} (idx_per_ii=1)
{
  GRANT_ONCE, [$0] -> [WEST, RED] (t=2, inv_iters=0)
} (idx_per_ii=2)

PE(0,1):
{
  PHI_START, [$0], [SOUTH, RED] -> [$0], [SOUTH, RED] (t=4, inv_iters=1)
} (idx_per_ii=0)
{
  FADD, [NORTH, RED], [$0] -> [NORTH, RED], [EAST, RED] (t=5, inv_iters=1)
} (idx_per_ii=1)
{
  CONSTANT, [#3.000000] -> [$0] (t=2, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [SOUTH, RED] (t=6, inv_iters=1)
} (idx_per_ii=2)
{
  GRANT_ONCE, [$0] -> [$0] (t=3, inv_iters=0)
} (idx_per_ii=3)

PE(1,1):
{
  CONSTANT, [#1] -> [$0] (t=0, inv_iters=0)
} (idx_per_ii=0)
{
  GRANT_ONCE, [$0] -> [$0] (t=1, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [WEST, RED] (t=5, inv_iters=1)
} (idx_per_ii=1)
{
  PHI_START, [$0], [EAST, RED] -> [$0], [EAST, RED] (t=2, inv_iters=0)
  DATA_MOV, [WEST, RED] -> [NORTH, RED] (t=6, inv_iters=1)
} (idx_per_ii=2)
{
  ADD, [SOUTH, RED], [$0] -> [EAST, RED], [SOUTH, RED] (t=3, inv_iters=0)
  DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=7, inv_iters=1)
} (idx_per_ii=3)

PE(2,1):
{
  ICMP_SLT, [WEST, RED], [SOUTH, RED] -> [SOUTH, RED], [NORTH, RED], [WEST, RED], [$2], [$1], [EAST, RED] (t=4, inv_iters=1)
} (idx_per_ii=0)
{
  GRANT_PREDICATE, [$0], [$2] -> [WEST, RED] (t=5, inv_iters=1)
} (idx_per_ii=1)
{
  GRANT_PREDICATE, [$0], [$1] -> [SOUTH, RED] (t=6, inv_iters=1)
  DATA_MOV, [EAST, RED] -> [WEST, RED] (t=6, inv_iters=1)
} (idx_per_ii=2)
{
  DATA_MOV, [WEST, RED] -> [$0] (t=3, inv_iters=0)
} (idx_per_ii=3)

PE(3,1):
{
  NOT, [WEST, RED] -> [WEST, RED] (t=5, inv_iters=1)
} (idx_per_ii=1)

PE(0,2):
{
  PHI_START, [NORTH, RED], [EAST, RED] -> [SOUTH, RED] (t=4, inv_iters=1)
} (idx_per_ii=0)
{
  DATA_MOV, [SOUTH, RED] -> [EAST, RED] (t=6, inv_iters=1)
} (idx_per_ii=2)

PE(1,2):
{
  GRANT_PREDICATE, [$0], [SOUTH, RED] -> [$0] (t=8, inv_iters=2)
} (idx_per_ii=0)
{
  RETURN, [$0] (t=9, inv_iters=2)
} (idx_per_ii=1)
{
  DATA_MOV, [EAST, RED] -> [$0] (t=6, inv_iters=1)
} (idx_per_ii=2)
{
  GRANT_PREDICATE, [WEST, RED], [$0] -> [WEST, RED] (t=7, inv_iters=1)
  DATA_MOV, [SOUTH, RED] -> [$0] (t=7, inv_iters=1)
} (idx_per_ii=3)

PE(2,2):
{
  DATA_MOV, [SOUTH, RED] -> [WEST, RED] (t=5, inv_iters=1)
} (idx_per_ii=1)

PE(0,3):
{
  CONSTANT, [#0.000000] -> [$0] (t=2, inv_iters=0)
} (idx_per_ii=2)
{
  GRANT_ONCE, [$0] -> [SOUTH, RED] (t=3, inv_iters=0)
} (idx_per_ii=3)

