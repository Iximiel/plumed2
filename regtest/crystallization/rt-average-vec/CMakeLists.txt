PLUMED_TEST(rt-average-vec TYPE driver
ARGS "--plumed plumed.dat --ixyz LJ75C_global-minima.xyz"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
# this is to test a different name
arg="--plumed plumed.dat --ixyz LJ75C_global-minima.xyz"
]] )
