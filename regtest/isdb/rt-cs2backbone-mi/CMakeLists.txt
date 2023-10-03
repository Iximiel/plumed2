PLUMED_TEST(rt-cs2backbone-mi TYPE driver
ARGS "--plumed plumed.dat --mf_pdb traj.pdb --multi 2 --dump-forces atom_forces"
MPIPROCS 4
NEEDS cregex
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs=cregex
mpiprocs=4
type=driver
arg="--plumed plumed.dat --mf_pdb traj.pdb --multi 2 --dump-forces atom_forces"
]] )
