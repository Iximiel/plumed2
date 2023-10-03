PLUMED_TEST(rt-fametad-1 TYPE driver
ARGS "--plumed plumed.dat --noatoms --timestep 1.0"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --noatoms --timestep 1.0"
]] )
