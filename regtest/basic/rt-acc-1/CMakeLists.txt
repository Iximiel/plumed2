PLUMED_TEST(rt-acc-1 TYPE driver
ARGS "--plumed plumed.dat --timestep 1.0 --trajectory-stride 0 --ixtc aladip.xtc"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --timestep 1.0 --trajectory-stride 0 --ixtc aladip.xtc"
]] )
