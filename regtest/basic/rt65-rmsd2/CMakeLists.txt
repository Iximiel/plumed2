PLUMED_TEST(NAME rt65-rmsd2 TYPE make
EXTRAFILES ../../trajectories/rmsd/1GB1_mdl1.pdb  ../../trajectories/rmsd/1GB1_mdl2.pdb ../../trajectories/rmsd/1GB1_mdl1_rototranslated.pdb
)
#the following variable is the original config file
set(originalConfig [[
type=make
extra_files="../../trajectories/rmsd/1GB1_mdl1.pdb  ../../trajectories/rmsd/1GB1_mdl2.pdb ../../trajectories/rmsd/1GB1_mdl1_rototranslated.pdb"
#arg=" 0 1 2 "
]] )
