PLUMED_TEST(NAME rt47 TYPE driver
ARGS "--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed=plumed.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff"
]] )
