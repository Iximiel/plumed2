PLUMED_TEST(rt-dfg2 TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz --length-units A --box 28.66,28.66,28.66"
MPIPROCS 2
MODULES adjmat
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
plumed_modules=adjmat
# this is to test a different name
arg="--plumed plumed.dat --ixyz traj.xyz --length-units A --box 28.66,28.66,28.66"
]] )
