PLUMED_TEST(rt-pamm-aperiodic TYPE driver
ARGS "--plumed plumed.dat --ixyz ncub.xyz"
MODULES pamm
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=pamm
# this is to test a different name
arg="--plumed plumed.dat --ixyz ncub.xyz"
]] )
