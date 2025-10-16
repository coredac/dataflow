PE(0,0):
{
  GRANT_ONCE, [NORTH, RED] -> [$0]
} (t=5)
{
  PHI, [NORTH, RED], [$0] -> [EAST, RED], [NORTH, RED]
} (t=6)
{
  CONSTANT, [%arg0, RED]
} (t=8)
{
  CTRL_MOV, [SOUTH, RED] -> [$1]
} (t=11)

PE(1,0):
{
  DATA_MOV, [WEST, RED] -> [$32]
} (t=4)
{
  DATA_MOV, [SOUTH, RED] -> [$33]
} (t=6)
{
  DATA_MOV, [WEST, RED] -> [NORTH, RED]
} (t=7)
{
  GRANT_PREDICATE, [$32], [$33] -> [EAST, RED]
} (t=8)

PE(2,0):
{
  DATA_MOV, [NORTH, RED] -> [WEST, RED]
} (t=3)
{
  RETURN
} (t=8)
{
  CTRL_MOV, [WEST, RED] -> [NORTH, RED]
} (t=9)

PE(0,1):
{
  GRANT_ONCE, [NORTH, RED] -> [$128]
} (t=1)
{
  PHI, [EAST, RED], [$128] -> [EAST, RED], [NORTH, RED], [$128]
} (t=2)
{
  ADD, [$128] -> [EAST, RED]
} (t=3)
{
  CONSTANT, [%arg1, RED] -> [SOUTH, RED]
} (t=4)
{
  DATA_MOV, [WEST, RED] -> [$128]
} (t=5)
{
  DATA_MOV, [EAST, RED] -> [EAST, RED]
} (t=6)
{
  GRANT_PREDICATE, [SOUTH, RED], [$128] -> [NORTH, RED]
  CTRL_MOV, [WEST, RED] -> [$129]
} (t=7)
{
  CTRL_MOV, [EAST, RED] -> [SOUTH, RED]
} (t=10)

PE(1,1):
{
  GEP, [EAST, RED], [WEST, RED] -> [EAST, RED]
  DATA_MOV, [EAST, RED] -> [$160]
} (t=3)
{
  ICMP, [WEST, RED] -> [$160], [NORTH, RED], [WEST, RED]
} (t=4)
{
  NOT, [$160] -> [$161], [WEST, RED], [SOUTH, RED], [NORTH, RED], [$162]
  DATA_MOV, [EAST, RED] -> [NORTH, RED]
} (t=5)
{
  GRANT_PREDICATE, [$160], [$161] -> [WEST, RED]
} (t=6)
{
  DATA_MOV, [WEST, RED] -> [NORTH, RED]
} (t=7)
{
  CONSTANT, [%arg1, RED]
  DATA_MOV, [NORTH, RED] -> [$160]
} (t=8)
{
  GRANT_PREDICATE, [$160], [$162] -> [WEST, RED]
} (t=9)

PE(2,1):
{
  GRANT_ONCE, [NORTH, RED] -> [$192]
} (t=1)
{
  PHI, [SOUTH, RED], [$192] -> [WEST, RED], [SOUTH, RED]
} (t=2)
{
  LOAD, [WEST, RED] -> [WEST, RED]
} (t=4)
{
  CONSTANT, [%arg2, RED]
} (t=8)

PE(0,2):
{
  CONSTANT, [#0] -> [SOUTH, RED]
} (t=0)
{
  DATA_MOV, [NORTH, RED] -> [$256]
} (t=3)
{
  GEP, [NORTH, RED], [$256] -> [$256]
} (t=4)
{
  LOAD, [$256] -> [EAST, RED]
} (t=5)
{
  DATA_MOV, [SOUTH, RED] -> [NORTH, RED]
} (t=8)

PE(1,2):
{
  DATA_MOV, [SOUTH, RED] -> [$290]
} (t=3)
{
  DATA_MOV, [NORTH, RED] -> [$288]
  DATA_MOV, [SOUTH, RED] -> [NORTH, RED]
} (t=5)
{
  FMUL, [WEST, RED], [SOUTH, RED] -> [$288]
  DATA_MOV, [WEST, RED] -> [$289]
} (t=6)
{
  FADD, [$288], [$289] -> [$289], [NORTH, RED]
} (t=7)
{
  GRANT_PREDICATE, [$290], [$288] -> [NORTH, RED]
  DATA_MOV, [NORTH, RED] -> [$288]
} (t=8)
{
  GRANT_PREDICATE, [$289], [$288] -> [EAST, RED]
} (t=9)

PE(2,2):
{
  CONSTANT, [%arg0, RED] -> [SOUTH, RED]
} (t=0)
{
  CONSTANT, [#0.000000] -> [EAST, RED]
} (t=3)
{
  PHI, [WEST, RED], [EAST, RED] -> [WEST, RED]
} (t=5)
{
  CTRL_MOV, [EAST, RED] -> [$320]
} (t=10)

PE(3,2):
{
  GRANT_ONCE, [WEST, RED] -> [WEST, RED]
} (t=4)

PE(0,3):
{
  DATA_MOV, [EAST, RED] -> [SOUTH, RED]
} (t=3)
{
  STORE, [EAST, RED], [SOUTH, RED]
} (t=9)

PE(1,3):
{
  GRANT_ONCE, [EAST, RED] -> [$416]
} (t=1)
{
  PHI, [SOUTH, RED], [$416] -> [WEST, RED], [SOUTH, RED]
} (t=2)
{
  DATA_MOV, [NORTH, RED] -> [$416]
} (t=6)
{
  GRANT_PREDICATE, [SOUTH, RED], [$416] -> [WEST, RED]
} (t=8)
{
  CTRL_MOV, [NORTH, RED] -> [$417]
} (t=9)

PE(2,3):
{
  CONSTANT, [%arg2, RED] -> [WEST, RED]
} (t=0)

