PLUMED_TEST(rt-smac TYPE driver
ARGS "--plumed plumed.dat --ixyz forgareth.xyz --length-units=A --box=70.0,70.0,70.0"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
# this is to test a different name
arg="--plumed plumed.dat --ixyz forgareth.xyz --length-units=A --box=70.0,70.0,70.0"
]] )
