PLUMED_TEST(rt-ttmetad2 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 500 --timestep 0.002 --igro traj.gro"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 500 --timestep 0.002 --igro traj.gro"
]] )
