PLUMED_TEST(rt-dimer-0 TYPE driver
ARGS "--plumed plumed.dat --mf_pdb aladip.pdb"
EXTRAFILES ../../trajectories/dimer/aladip.pdb
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --mf_pdb aladip.pdb"
extra_files="../../trajectories/dimer/aladip.pdb"
]] )
