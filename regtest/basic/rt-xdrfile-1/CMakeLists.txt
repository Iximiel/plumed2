PLUMED_TEST(NAME rt-xdrfile-1 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 0 --timestep 0.005 --itrr traj.trr"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 0 --timestep 0.005 --itrr traj.trr"
]] )
