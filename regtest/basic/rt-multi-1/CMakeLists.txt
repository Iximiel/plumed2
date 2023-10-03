PLUMED_TEST(rt-multi-1 TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz --multi 3 --debug-forces ff"
MPIPROCS 6
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=6
type=driver
arg="--plumed plumed.dat --ixyz traj.xyz --multi 3 --debug-forces ff"
]] )
