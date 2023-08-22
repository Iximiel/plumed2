PLUMED_TEST(NAME rt-molfile-4 TYPE driver
ARGS "--plumed plumed.dat --mf_crd test.crd --natoms 2257"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --mf_crd test.crd --natoms 2257"
]] )
