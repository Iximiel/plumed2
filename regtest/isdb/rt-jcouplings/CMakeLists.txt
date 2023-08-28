PLUMED_TEST(rt-jcouplings TYPE driver
ARGS "--plumed plumed.dat --mf_pdb peptide.pdb"
NEEDS cregex
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs="cregex"
type=driver
# this is to test a different name
arg="--plumed plumed.dat --mf_pdb peptide.pdb"
]] )
