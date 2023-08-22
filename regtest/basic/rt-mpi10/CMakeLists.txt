PLUMED_TEST(NAME rt-mpi10 TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3 --debug-dd yes --debug-grex 6 --debug-grex-log log.grex --initial-step 3"
MPIPROCS 6
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=6
type=driver
arg="--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3 --debug-dd yes --debug-grex 6 --debug-grex-log log.grex --initial-step 3"
]] )
