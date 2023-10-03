PLUMED_TEST(rt-mpi6b TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3"
MPIPROCS 3
)
#the following variable is the original config file
set(originalConfig [[
# this test duplicates test-mpi6
# but uses 6 mpi procs and 2 threads (default on travis)
mpiprocs=3
type=driver
# this is to test a different name
arg="--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3"
]] )
