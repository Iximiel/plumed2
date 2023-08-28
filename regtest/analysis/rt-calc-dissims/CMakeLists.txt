PLUMED_TEST(rt-calc-dissims TYPE driver
ARGS "--noatoms --plumed plumed.dat"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--noatoms --plumed plumed.dat"

]] )
