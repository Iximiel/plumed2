PLUMED_TEST(rt66dna TYPE driver
ARGS "--igro 1bna.gro"
NEEDS cregex
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs="cregex"
type=driver
arg="--igro 1bna.gro"
]] )
