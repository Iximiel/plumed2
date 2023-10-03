PLUMED_TEST(rt-piv-derivatives TYPE driver
ARGS "--plumed plumed.dat --mf_pdb Ice8.pdb"
MODULES piv
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=piv
arg="--plumed plumed.dat --mf_pdb Ice8.pdb"

]] )
