PLUMED_TEST(rt-pamm-periodic TYPE driver
ARGS "--plumed plumed.dat --ixyz M1d_open.xyz --box 55.598,55.569,55.569 --length-units A"
MODULES pamm
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=pamm
# this is to test a different name
arg="--plumed plumed.dat --ixyz M1d_open.xyz --box 55.598,55.569,55.569 --length-units A"
]] )
