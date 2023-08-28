PLUMED_TEST(rt-bf-legendre-scaled TYPE driver
ARGS "--plumed plumed.dat --noatoms"
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
type=driver
arg="--plumed plumed.dat --noatoms"

]] )
