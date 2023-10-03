PLUMED_TEST(rt-le-1d-outside TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 500 --timestep 0.002 --igro traj.gro --dump-forces forces --dump-forces-fmt=%10.6f "
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
type=driver
arg="--plumed plumed.dat --trajectory-stride 500 --timestep 0.002 --igro traj.gro --dump-forces forces --dump-forces-fmt=%10.6f "
]] )
