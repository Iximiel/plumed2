PLUMED_TEST(NAME rt-plumed-mpi TYPE driver
ARGS "--plumed=plumedx.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff  --debug-pd yes"
MPIPROCS 4
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
# this is to test a different name
arg="--plumed=plumedx.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff  --debug-pd yes"
extra_files="../../trajectories/trajectory.xyz"
]] )
