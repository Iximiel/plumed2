PLUMED_TEST(rt-fametad-2-restart TYPE driver
ARGS "--plumed plumed.dat --noatoms --timestep 1.0  --initial-step 101"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --noatoms --timestep 1.0  --initial-step 101"
]] )
