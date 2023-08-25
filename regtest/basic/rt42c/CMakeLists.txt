PLUMED_TEST(rt42c TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --ixyz trajectory.xyz"
]] )
