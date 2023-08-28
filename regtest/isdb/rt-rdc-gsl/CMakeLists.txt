PLUMED_TEST(rt-rdc-gsl TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz"
NEEDS gsl
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_needs="gsl"
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz"
]] )
