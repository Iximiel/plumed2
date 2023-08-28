PLUMED_TEST(rt-wcsurface2 TYPE driver
ARGS "--plumed plumed.dat --ixyz traj-3frames.xyz"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
# this is to test a different name
arg="--plumed plumed.dat --ixyz traj-3frames.xyz"
]] )
