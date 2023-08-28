PLUMED_TEST(rt-sel TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz"
MPIPROCS 1
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=1
type=driver
arg="--plumed plumed.dat --ixyz traj.xyz"
]] )
