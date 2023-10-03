PLUMED_TEST(rt-hbpamm TYPE driver
ARGS "--plumed plumed.dat --ixyz b3lyp.xyz"
MODULES pamm
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=pamm
# this is to test a different name
arg="--plumed plumed.dat --ixyz b3lyp.xyz"
]] )
