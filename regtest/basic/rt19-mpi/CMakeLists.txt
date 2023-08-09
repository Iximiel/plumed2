PLUMED_TEST(NAME rt19-mpi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%8.4f --multi 2"
MPIPROCS 2
NEEDS zlib
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
plumed_needs=zlib
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%8.4f --multi 2"
extra_files="../../trajectories/trajectory.xyz"

function plumed_regtest_after(){
  gunzip *.gz
}

]] )
