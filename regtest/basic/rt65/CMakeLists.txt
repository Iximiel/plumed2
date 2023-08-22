PLUMED_TEST(NAME rt65 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.002 --ixyz trajectory.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.002 --ixyz trajectory.xyz"
]] )
