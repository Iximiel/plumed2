PLUMED_TEST(rt-rdc-mi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz ala12_trajectory.xyz --multi 2  --dump-forces allforce"
MPIPROCS 4
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz ala12_trajectory.xyz --multi 2  --dump-forces allforce"
]] )
