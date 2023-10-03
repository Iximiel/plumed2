PLUMED_TEST(rt-env-sim-deriv TYPE driver
ARGS "--plumed plumed.dat --mf_pdb bcc.pdb"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
# this is to test a different name
arg="--plumed plumed.dat --mf_pdb bcc.pdb"
]] )
