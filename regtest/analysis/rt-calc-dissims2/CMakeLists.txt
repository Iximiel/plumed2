PLUMED_TEST(rt-calc-dissims2 TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz"
EXTRAFILES ../rt-read-dissims2/traj.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --ixyz traj.xyz"
extra_files="../rt-read-dissims2/traj.xyz"
]] )
