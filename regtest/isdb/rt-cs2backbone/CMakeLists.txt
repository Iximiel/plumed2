PLUMED_TEST(rt-cs2backbone TYPE driver
ARGS "--plumed plumed.dat --mf_pdb traj.pdb --dump-forces atom_forces"
NEEDS cregex
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs="cregex"
type=driver
arg="--plumed plumed.dat --mf_pdb traj.pdb --dump-forces atom_forces"
]] )
