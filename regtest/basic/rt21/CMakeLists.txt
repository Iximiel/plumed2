PLUMED_TEST(rt21 TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --ixyz traj.xyz"
]] )
