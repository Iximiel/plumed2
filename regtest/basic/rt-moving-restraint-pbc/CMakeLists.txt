PLUMED_TEST(NAME rt-moving-restraint-pbc TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --ixyz traj.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
