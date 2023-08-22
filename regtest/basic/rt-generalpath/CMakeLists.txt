PLUMED_TEST(NAME rt-generalpath TYPE driver
ARGS "--plumed path_test.dat --noatoms --timestep 2"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed path_test.dat --noatoms --timestep 2"
]] )
