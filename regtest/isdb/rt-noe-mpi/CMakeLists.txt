PLUMED_TEST(rt-noe-mpi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz ala12_trajectory.xyz --multi 2"
MPIPROCS 2
NEEDS cregex
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs=cregex
mpiprocs=2
type=driver
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz ala12_trajectory.xyz --multi 2"
]] )
