PLUMED_TEST(rt-atomic-units TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.0005 --ixyz coord.xyz --length-units A"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.0005 --ixyz coord.xyz --length-units A"
]] )
