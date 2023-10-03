PLUMED_TEST(rt-drift-2 TYPE driver
ARGS "--plumed plumed.dat --timestep=0.005  --initial-step=1 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --debug-dd yes"
MPIPROCS 4
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
# this is to test a different name
arg="--plumed plumed.dat --timestep=0.005  --initial-step=1 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --debug-dd yes"

]] )
