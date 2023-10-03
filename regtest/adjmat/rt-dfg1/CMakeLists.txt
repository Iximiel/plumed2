PLUMED_TEST(rt-dfg1 TYPE driver
ARGS "--plumed plumed.dat --ixyz config.xyz"
MPIPROCS 4
MODULES adjmat
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
plumed_modules=adjmat
# this is to test a different name
arg="--plumed plumed.dat --ixyz config.xyz"
]] )
