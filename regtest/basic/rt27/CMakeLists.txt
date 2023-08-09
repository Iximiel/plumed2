PLUMED_TEST(NAME rt27 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
extra_files="../../trajectories/trajectory.xyz"
]] )
