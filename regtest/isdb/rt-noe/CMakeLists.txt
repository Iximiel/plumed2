PLUMED_TEST(rt-noe TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz ala12_trajectory.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz ala12_trajectory.xyz"
]] )
