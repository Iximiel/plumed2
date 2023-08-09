PLUMED_TEST(NAME rt-molfile-6 TYPE driver
ARGS "--plumed plumed.dat --mf_pdb input.pdb"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --mf_pdb input.pdb"
]] )
