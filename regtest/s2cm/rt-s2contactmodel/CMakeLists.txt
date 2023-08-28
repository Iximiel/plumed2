PLUMED_TEST(rt-s2contactmodel TYPE driver
ARGS "--plumed plumed.dat --mf_pdb Ching2.pdb --box 100.0,100.0,100.0"
MODULES s2cm
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=s2cm
type=driver
# this is to test a different name
arg="--plumed plumed.dat --mf_pdb Ching2.pdb --box 100.0,100.0,100.0"
]] )
