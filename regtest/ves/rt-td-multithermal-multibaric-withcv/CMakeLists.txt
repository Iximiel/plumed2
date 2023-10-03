PLUMED_TEST(rt-td-multithermal-multibaric-withcv TYPE driver
ARGS "--plumed plumed.dat --noatoms"
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
type=driver
arg="--plumed plumed.dat --noatoms"

]] )
