PE(0,0):
{
  CONSTANT, [#1] -> [$3], [$0], [$2], [EAST, RED], [NORTH, RED], [$1]
} (t=0)
{
  CONSTANT, [#32] -> [NORTH, RED]
} (t=2)
{
  ALLOCA, [$0] -> [NORTH, RED], [EAST, RED]
} (t=4)
{
  ALLOCA, [$1] -> [NORTH, RED], [EAST, RED]
} (t=6)
{
  ALLOCA, [$2] -> [EAST, RED], [$6]
} (t=7)
{
  GRANT_ONCE, [$3] -> [$0]
} (t=8)
{
  PHI, [EAST, RED], [$0] -> [EAST, RED]
} (t=9)
{
  CTRL_MOV, [WEST, RED] -> [$4]
} (t=11)
{
  DATA_MOV, [EAST, RED] -> [NORTH, RED]
} (t=12)
{
  DATA_MOV, [WEST, RED] -> [$5]
} (t=13)
{
  STORE, [$5], [$6]
} (t=14)

PE(1,0):
{
  DATA_MOV, [WEST, RED] -> [EAST, RED]
} (t=1)
{
  DATA_MOV, [WEST, RED] -> [EAST, RED]
  DATA_MOV, [EAST, RED] -> [NORTH, RED]
  DATA_MOV, [SOUTH, RED] -> [$34]
} (t=5)
{
  DATA_MOV, [SOUTH, RED] -> [$33]
} (t=6)
{
  LOAD, [NORTH, RED] -> [EAST, RED]
  DATA_MOV, [WEST, RED] -> [NORTH, RED]
} (t=7)
{
  GRANT_ONCE, [WEST, RED] -> [$32]
} (t=8)
{
  PHI, [$35], [$32] -> [$32], [EAST, RED]
  DATA_MOV, [EAST, RED] -> [NORTH, RED]
} (t=9)
{
  GRANT_PREDICATE, [WEST, RED], [$33] -> [EAST, RED], [WEST, RED]
} (t=10)
{
  LOAD, [NORTH, RED] -> [WEST, RED]
} (t=11)
{
  GRANT_PREDICATE, [$32], [$34] -> [$35]
  DATA_MOV, [NORTH, RED] -> [WEST, RED]
} (t=12)
{
  DATA_MOV, [WEST, RED] -> [$32]
} (t=13)
{
  STORE, [NORTH, RED], [$32]
} (t=14)

PE(2,0):
{
  DATA_MOV, [EAST, RED] -> [$64]
} (t=2)
{
  ALLOCA, [$64] -> [EAST, RED], [WEST, RED]
} (t=4)
{
  DATA_MOV, [WEST, RED] -> [EAST, RED]
} (t=6)
{
  DATA_MOV, [EAST, RED] -> [NORTH, RED]
} (t=7)
{
  SEXT, [WEST, RED] -> [WEST, RED]
} (t=8)
{
  DATA_MOV, [WEST, RED] -> [NORTH, RED]
} (t=10)
{
  DATA_MOV, [WEST, RED] -> [NORTH, RED]
} (t=11)
{
  LOAD, [NORTH, RED] -> [WEST, RED]
} (t=12)

PE(3,0):
{
  GRANT_ONCE, [WEST, RED] -> [$96]
} (t=5)
{
  PHI, [NORTH, RED], [$96] -> [WEST, RED]
} (t=6)
{
  DATA_MOV, [EAST, RED] -> [$96]
} (t=7)
{
  CTRL_MOV, [SOUTH, RED] -> [$97]
} (t=10)
{
  STORE, [NORTH, RED], [$96]
} (t=12)

PE(0,1):
{
  ALLOCA, [SOUTH, RED] -> [EAST, RED], [NORTH, RED]
} (t=1)
{
  GRANT_ONCE, [SOUTH, RED] -> [$128]
} (t=3)
{
  PHI, [EAST, RED], [$128] -> [EAST, RED]
} (t=4)
{
  GRANT_ONCE, [SOUTH, RED] -> [$128]
} (t=5)
{
  PHI, [EAST, RED], [$128] -> [EAST, RED]
} (t=6)
{
  GRANT_ONCE, [SOUTH, RED] -> [$128]
} (t=7)
{
  PHI, [EAST, RED], [$128] -> [EAST, RED], [NORTH, RED]
  CTRL_MOV, [WEST, RED] -> [$129]
} (t=8)
{
  CTRL_MOV, [WEST, RED] -> [$130]
} (t=9)
{
  CTRL_MOV, [WEST, RED] -> [$131]
} (t=10)
{
  DATA_MOV, [SOUTH, RED] -> [$128]
  DATA_MOV, [WEST, RED] -> [$132]
} (t=12)
{
  FMUL_FADD, [SOUTH, RED], [$128], [$132] -> [NORTH, RED]
} (t=13)

PE(1,1):
{
  GRANT_ONCE, [WEST, RED] -> [$160]
} (t=2)
{
  PHI, [$162], [$160] -> [$160], [$162]
} (t=3)
{
  LOAD, [$160] -> [$160]
  DATA_MOV, [EAST, RED] -> [$160]
} (t=4)
{
  ICMP, [$160], [WEST, RED] -> [$163], [$160], [EAST, RED], [$164], [SOUTH, RED], [$161], [NORTH, RED]
} (t=5)
{
  GRANT_PREDICATE, [$162], [$160] -> [SOUTH, RED], [NORTH, RED], [EAST, RED], [$162]
  DATA_MOV, [NORTH, RED] -> [$165]
} (t=6)
{
  GRANT_PREDICATE, [$160], [$161] -> [WEST, RED]
  DATA_MOV, [EAST, RED] -> [$160]
} (t=7)
{
  GRANT_PREDICATE, [$160], [$163] -> [EAST, RED], [WEST, RED]
  DATA_MOV, [SOUTH, RED] -> [NORTH, RED]
} (t=8)
{
  GRANT_PREDICATE, [WEST, RED], [$164] -> [EAST, RED], [NORTH, RED], [WEST, RED]
} (t=9)
{
  GEP, [EAST, RED], [SOUTH, RED] -> [SOUTH, RED]
} (t=10)
{
  CONSTANT, [#0] -> [SOUTH, RED]
  DATA_MOV, [EAST, RED] -> [WEST, RED]
} (t=11)
{
  STORE, [NORTH, RED], [$165]
} (t=12)
{
  DATA_MOV, [NORTH, RED] -> [SOUTH, RED]
} (t=13)

PE(2,1):
{
  DATA_MOV, [EAST, RED] -> [$192]
} (t=6)
{
  DATA_MOV, [WEST, RED] -> [EAST, RED]
} (t=7)
{
  GRANT_PREDICATE, [SOUTH, RED], [$192] -> [NORTH, RED], [EAST, RED]
} (t=8)
{
  LOAD, [WEST, RED] -> [WEST, RED]
} (t=9)
{
  LOAD, [WEST, RED] -> [WEST, RED]
} (t=10)
{
  GRANT_PREDICATE, [SOUTH, RED], [NORTH, RED] -> [SOUTH, RED]
  DATA_MOV, [WEST, RED] -> [$192]
} (t=11)
{
  ADD, [$192], [SOUTH, RED] -> [NORTH, RED]
} (t=12)

PE(3,1):
{
  DATA_MOV, [EAST, RED] -> [$224]
} (t=8)
{
  CTRL_MOV, [WEST, RED] -> [SOUTH, RED]
} (t=9)
{
  LOAD, [$224] -> [WEST, RED]
} (t=10)
{
  CONSTANT, [#0] -> [SOUTH, RED]
} (t=11)

PE(0,2):
{
  DATA_MOV, [SOUTH, RED] -> [EAST, RED]
} (t=2)
{
  SEXT, [EAST, RED] -> [EAST, RED]
} (t=8)
{
  DATA_MOV, [SOUTH, RED] -> [EAST, RED]
} (t=9)
{
  LOAD, [EAST, RED] -> [SOUTH, RED]
} (t=11)
{
  DATA_MOV, [EAST, RED] -> [NORTH, RED]
} (t=12)
{
  DATA_MOV, [SOUTH, RED] -> [NORTH, RED]
} (t=14)

PE(1,2):
{
  DATA_MOV, [WEST, RED] -> [EAST, RED]
} (t=3)
{
  DATA_MOV, [SOUTH, RED] -> [EAST, RED]
} (t=6)
{
  LOAD, [SOUTH, RED] -> [WEST, RED]
} (t=7)
{
  DATA_MOV, [SOUTH, RED] -> [EAST, RED]
} (t=8)
{
  DATA_MOV, [SOUTH, RED] -> [EAST, RED]
  DATA_MOV, [EAST, RED] -> [$288]
} (t=9)
{
  GEP, [EAST, RED], [$288] -> [WEST, RED]
  DATA_MOV, [WEST, RED] -> [NORTH, RED]
} (t=10)
{
  CONSTANT, [#0] -> [SOUTH, RED]
  DATA_MOV, [SOUTH, RED] -> [WEST, RED]
} (t=11)
{
  LOAD, [NORTH, RED] -> [SOUTH, RED]
} (t=12)

PE(2,2):
{
  DATA_MOV, [WEST, RED] -> [NORTH, RED]
} (t=4)
{
  DATA_MOV, [EAST, RED] -> [$321]
} (t=6)
{
  DATA_MOV, [WEST, RED] -> [NORTH, RED]
} (t=7)
{
  LOAD, [SOUTH, RED] -> [WEST, RED]
} (t=9)
{
  DATA_MOV, [EAST, RED] -> [$320]
  DATA_MOV, [NORTH, RED] -> [SOUTH, RED]
} (t=10)
{
  STORE, [EAST, RED], [$320]
} (t=12)
{
  STORE, [SOUTH, RED], [$321]
} (t=13)

PE(3,2):
{
  CONSTANT, [#0.000000] -> [WEST, RED]
} (t=11)
{
  RETURN
} (t=12)

PE(0,3):
{
  DATA_MOV, [NORTH, RED] -> [$384]
} (t=10)
{
  STORE, [SOUTH, RED], [$384]
} (t=15)

PE(1,3):
{
  DATA_MOV, [WEST, RED] -> [$416]
} (t=10)
{
  GRANT_PREDICATE, [SOUTH, RED], [$416] -> [SOUTH, RED]
} (t=11)

PE(2,3):
{
  DATA_MOV, [NORTH, RED] -> [$449]
} (t=5)
{
  DATA_MOV, [NORTH, RED] -> [$448]
} (t=8)
{
  NOT, [$448] -> [WEST, RED], [SOUTH, RED]
} (t=9)
{
  STORE, [EAST, RED], [$449]
} (t=12)

PE(3,3):
{
  CONSTANT, [#0] -> [WEST, RED]
} (t=11)

