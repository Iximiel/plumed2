PLUMED_TEST(rt-smap-pointwise TYPE driver
ARGS "--plumed plumed.dat --noatoms"
MODULES dimred
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=dimred
# this is to test a different name
arg="--plumed plumed.dat --noatoms"
]] )
