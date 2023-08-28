PLUMED_TEST(rt-output-fes-2d-targetdist TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 500 --timestep 0.002 --igro conf.gro "
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
type=driver
arg="--plumed plumed.dat --trajectory-stride 500 --timestep 0.002 --igro conf.gro "
]] )
