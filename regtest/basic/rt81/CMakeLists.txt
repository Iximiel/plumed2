PLUMED_TEST(rt81 TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3"
MPIPROCS 6
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=6
type=driver
# this is to test a different name
arg="--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3"
# this is to enforce two threads
PLUMED_NUM_THREADS=1
]] )
