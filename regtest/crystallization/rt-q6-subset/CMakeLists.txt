PLUMED_TEST(rt-q6-subset TYPE driver
ARGS "--plumed plumed.dat --ixyz 64.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
# this is to test a different name
arg="--plumed plumed.dat --ixyz 64.xyz --dump-forces forces --dump-forces-fmt=%8.4f" # --debug-forces forces.num"
]] )
