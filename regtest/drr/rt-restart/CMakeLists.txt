PLUMED_TEST(rt-restart TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz ala12_trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
MODULES drr
NEEDS boost_serialization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_needs=boost_serialization
plumed_modules=drr
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz ala12_trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
]] )
