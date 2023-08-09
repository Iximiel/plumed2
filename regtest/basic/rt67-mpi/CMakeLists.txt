PLUMED_TEST(NAME rt67-mpi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 500 --timestep 0.002 --igro traj.gro"
MPIPROCS 4
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 500 --timestep 0.002 --igro traj.gro"
]] )
