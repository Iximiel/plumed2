PLUMED_TEST(rt-piv-distance TYPE driver
ARGS "--plumed plumed.dat --mf_pdb Ice.pdb"
MODULES piv
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=piv
arg="--plumed plumed.dat --mf_pdb Ice.pdb"

]] )
