PLUMED_TEST(NAME rt73 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz trajectory.xyz"
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz trajectory.xyz"
extra_files="../../trajectories/trajectory.xyz"
]] )
