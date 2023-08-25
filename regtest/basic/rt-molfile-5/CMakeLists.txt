PLUMED_TEST(rt-molfile-5 TYPE driver
ARGS "--plumed plumed.dat --mf_crdbox test.crd --natoms 9560"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --mf_crdbox test.crd --natoms 9560"
]] )
