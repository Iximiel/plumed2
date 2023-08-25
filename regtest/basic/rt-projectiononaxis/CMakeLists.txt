PLUMED_TEST(rt-projectiononaxis TYPE driver
ARGS "--plumed projection_test.dat --mf_xtc trial_path.xtc --pdb mc_file.pdb"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed projection_test.dat --mf_xtc trial_path.xtc --pdb mc_file.pdb"
]] )
