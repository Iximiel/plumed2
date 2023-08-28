PLUMED_TEST(rt-opt-memetic-sampling-heuristics TYPE driver
ARGS "--plumed plumed.dat --timestep 1 --igro trajectory.gro"
MODULES maze
EXTRAFILES ../traj/trajectory.gro
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=maze

arg="--plumed plumed.dat --timestep 1 --igro trajectory.gro"
extra_files="../traj/trajectory.gro"
]] )
