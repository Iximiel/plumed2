PLUMED_TEST(rt-fes-periodic TYPE driver
ARGS "--plumed plumed.dat --igro traj.gro"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --igro traj.gro"
]] )
