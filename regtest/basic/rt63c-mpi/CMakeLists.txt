PLUMED_TEST(rt63c-mpi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --igro helix.input.gro --dump-forces ff --dump-full-virial --dump-forces-fmt=%7.4f --debug-dd yes"
MPIPROCS 2
)
#the following variable is the original config file
set(originalConfig [[
type=driver
mpiprocs=2
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --igro helix.input.gro --dump-forces ff --dump-full-virial --dump-forces-fmt=%7.4f --debug-dd yes"
]] )
