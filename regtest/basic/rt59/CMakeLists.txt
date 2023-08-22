PLUMED_TEST(NAME rt59 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --igro traj_10dec.gro "
EXTRAFILES ../../trajectories/grofiles/traj_10dec.gro ../../trajectories/grofiles/sugar.dat 
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --igro traj_10dec.gro "
extra_files="../../trajectories/grofiles/traj_10dec.gro ../../trajectories/grofiles/sugar.dat "
]] )
