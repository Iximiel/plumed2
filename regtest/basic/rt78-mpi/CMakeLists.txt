PLUMED_TEST(NAME rt78-mpi TYPE driver
ARGS "--plumed plumed.dat --timestep 0.002 --mf_xtc trajectory.xtc --dump-forces forces --dump-forces-fmt=%.1f"
MPIPROCS 2
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
# this is to test a different name
arg="--plumed plumed.dat --timestep 0.002 --mf_xtc trajectory.xtc --dump-forces forces --dump-forces-fmt=%.1f"
]] )
