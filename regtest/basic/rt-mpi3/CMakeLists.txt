PLUMED_TEST(rt-mpi3 TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3
     --debug-dd yes --debug-grex 2 --debug-grex-log log.grex"
MPIPROCS 6
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=6
type=driver
# this is to test a different name
arg="--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3
     --debug-dd yes --debug-grex 2 --debug-grex-log log.grex"
]] )
