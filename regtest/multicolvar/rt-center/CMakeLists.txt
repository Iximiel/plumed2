PLUMED_TEST(rt-center TYPE driver
ARGS "--plumed plumed.dat --ixyz gas-one.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --ixyz gas-one.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
