PLUMED_TEST(rt-fix-228 TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz --multi 2"
MPIPROCS 2
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
# this is to test a different name
arg="--plumed plumed.dat --ixyz traj.xyz --multi 2"
]] )
