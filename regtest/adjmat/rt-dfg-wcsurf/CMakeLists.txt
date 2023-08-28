PLUMED_TEST(rt-dfg-wcsurf TYPE driver
ARGS "--plumed plumed.dat --ixyz gas-one.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
MODULES adjmat
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=adjmat
# this is to test a different name
arg="--plumed plumed.dat --ixyz gas-one.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
