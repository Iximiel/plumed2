PLUMED_TEST(rt-pmf1d TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
MODULES drr
NEEDS boost_serialization
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_needs=boost_serialization
plumed_modules=drr
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
extra_files="../../trajectories/trajectory.xyz"
]] )
