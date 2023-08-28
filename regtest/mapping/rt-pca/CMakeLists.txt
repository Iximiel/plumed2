PLUMED_TEST(rt-pca TYPE driver
ARGS "--plumed plumed.dat --length-units A --ixyz pca-test.xyz --box 100.,100.,100."
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --length-units A --ixyz pca-test.xyz --box 100.,100.,100."
]] )
