PLUMED_TEST(rt63-mpi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --igro helix.input.gro --dump-forces ff --dump-full-virial --debug-dd yes"
MPIPROCS 2
)
#the following variable is the original config file
set(originalConfig [[
type=driver
mpiprocs=2
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --igro helix.input.gro --dump-forces ff --dump-full-virial --debug-dd yes"
]] )
