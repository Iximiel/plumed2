PLUMED_TEST(NAME rt-mpi2 TYPE driver
ARGS "--debug-dd yes --plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb"
MPIPROCS 4
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
# this is to test a different name
arg="--debug-dd yes --plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb"
extra_files="../../trajectories/trajectory.xyz"
]] )
