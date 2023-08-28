PLUMED_TEST(rt-averaged-q6-spAspB TYPE driver
ARGS "--plumed plumed.dat --ixyz traj_md-10.xyz --length-units A --dump-forces forces --dump-forces-fmt=%8.4f"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
# this is to test a different name
arg="--plumed plumed.dat --ixyz traj_md-10.xyz --length-units A --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
