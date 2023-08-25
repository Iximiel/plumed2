PLUMED_TEST(rt-mpi7 TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --multi 3"
MPIPROCS 3
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=3
type=driver
# this is to test a different name
arg="--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --multi 3"
]] )
