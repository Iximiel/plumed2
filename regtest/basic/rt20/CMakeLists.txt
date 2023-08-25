PLUMED_TEST(rt20 TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --ixyz trajectory.xyz"
]] )
