PLUMED_TEST(rt-eds-adaptive TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --ixyz trajectory.xyz"
MODULES eds
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=eds
arg="--plumed plumed.dat --trajectory-stride 1 --ixyz trajectory.xyz"
extra_files="../../trajectories/trajectory.xyz"

]] )
