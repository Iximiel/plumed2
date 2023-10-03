PLUMED_TEST(rt42b TYPE driver
ARGS "--plumed plumed.dat --igro trajectory.gro"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --igro trajectory.gro"
]] )
