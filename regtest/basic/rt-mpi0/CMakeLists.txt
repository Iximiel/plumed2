PLUMED_TEST(NAME rt-mpi0 TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff"
MPIPROCS 4
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
# this is to test a different name
arg="--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff"
extra_files="../../trajectories/trajectory.xyz"
]] )
