PLUMED_TEST(rt15 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz"
NEEDS dlopen
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_needs="dlopen"
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz"
]] )
