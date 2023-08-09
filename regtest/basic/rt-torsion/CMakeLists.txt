PLUMED_TEST(NAME rt-torsion TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --ixyz traj.xyz"
]] )
