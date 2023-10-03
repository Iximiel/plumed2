PLUMED_TEST(rt54 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --igro helix.input.gro"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --igro helix.input.gro"
]] )
