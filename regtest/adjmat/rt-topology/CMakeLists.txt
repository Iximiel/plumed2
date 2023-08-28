PLUMED_TEST(rt-topology TYPE driver
ARGS " --plumed plumed.dat --ixyz frame_no_373.xyz --length-units 0.1"
MPIPROCS 4
MODULES adjmat
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
plumed_modules=adjmat
# this is to test a different name
arg=" --plumed plumed.dat --ixyz frame_no_373.xyz --length-units 0.1"
]] )
