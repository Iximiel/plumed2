PLUMED_TEST(rt-store-histo TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz"
MODULES dimred
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=dimred
# this is to test a different name
arg="--plumed plumed.dat --ixyz trajectory.xyz"
]] )
