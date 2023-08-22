PLUMED_TEST(NAME rt41 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 50 --timestep 0.005 --noatoms"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 50 --timestep 0.005 --noatoms"
]] )
