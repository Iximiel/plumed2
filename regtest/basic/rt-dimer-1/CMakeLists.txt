PLUMED_TEST(NAME rt-dimer-1 TYPE driver
ARGS "--plumed plumed.dat --mf_pdb aladip_wsol.pdb"
EXTRAFILES ../../trajectories/dimer/aladip_wsol.pdb
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --mf_pdb aladip_wsol.pdb"
extra_files="../../trajectories/dimer/aladip_wsol.pdb"
]] )
