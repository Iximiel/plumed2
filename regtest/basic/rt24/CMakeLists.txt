PLUMED_TEST(rt24 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz ala12_trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --ixyz ala12_trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
]] )
