PLUMED_TEST(NAME rt-numder2 TYPE driver
ARGS "--plumed plumed.dat --igro frame.gro --pdb masses.pdb --dump-forces-fmt %8.4f --debug-forces forces.dat"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --igro frame.gro --pdb masses.pdb --dump-forces-fmt %8.4f --debug-forces forces.dat"
]] )
