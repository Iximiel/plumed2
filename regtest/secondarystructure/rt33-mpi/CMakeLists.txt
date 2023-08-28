PLUMED_TEST(rt33-mpi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz amyloid.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
MPIPROCS 2
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz amyloid.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
]] )
