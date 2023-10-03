PLUMED_TEST(rt46 TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz amyloid.xyz --dump-forces ff"
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed=plumed.dat --timestep=0.05 --ixyz amyloid.xyz --dump-forces ff"
extra_files="../../trajectories/trajectory.xyz"
]] )
