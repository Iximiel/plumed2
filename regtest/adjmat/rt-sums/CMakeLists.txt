PLUMED_TEST(rt-sums TYPE driver
ARGS "--plumed plumed.dat --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
MODULES adjmat
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=adjmat
# this is to test a different name
arg="--plumed plumed.dat --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
extra_files="../../trajectories/trajectory.xyz"
]] )
