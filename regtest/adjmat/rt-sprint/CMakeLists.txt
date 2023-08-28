PLUMED_TEST(rt-sprint TYPE driver
ARGS "--plumed plumed.dat --ixyz isomers.xyz"
MODULES adjmat
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=adjmat
# this is to test a different name
arg="--plumed plumed.dat --ixyz isomers.xyz"
]] )
