PLUMED_TEST(rt58 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --igro traj_5dec.gro "
EXTRAFILES ../../trajectories/grofiles/traj_5dec.gro ../../trajectories/grofiles/sugar.dat 
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --igro traj_5dec.gro "
extra_files="../../trajectories/grofiles/traj_5dec.gro ../../trajectories/grofiles/sugar.dat "
]] )
