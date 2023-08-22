PLUMED_TEST(NAME rt10 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f --mc mc
  --box 5.0388,5.0388,5.0388"
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f --mc mc
  --box 5.0388,5.0388,5.0388"
extra_files="../../trajectories/trajectory.xyz"
]] )
