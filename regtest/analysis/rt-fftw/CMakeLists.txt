PLUMED_TEST(rt-fftw TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz"
NEEDS fftw
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_needs="fftw"
# this is to test a different name
arg="--plumed plumed.dat --ixyz trajectory.xyz"
]] )
