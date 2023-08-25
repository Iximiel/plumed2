PLUMED_TEST(rt70 TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --ixyz trajectory.xyz"
]] )
