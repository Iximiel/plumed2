PLUMED_TEST(rt-mpi2sync TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --debug-dd yes"
MPIPROCS 4
EXPORTVARIABLES
PLUMED_ASYNC_SHARE "no"
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
# this is to test a different name
arg="--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --debug-dd yes"
extra_files="../../trajectories/trajectory.xyz"
export PLUMED_ASYNC_SHARE=no
]] )
