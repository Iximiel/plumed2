PLUMED_TEST(rt-dist-contour TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is tc test a different name
arg="--plumed plumed.dat --ixyz trajectory.xyz"
]] )
