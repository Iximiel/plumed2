PLUMED_TEST(rt-maxent-3 TYPE driver
ARGS "--plumed plumed.dat --mf_xtc traj.xtc --dump-forces forces --multi 6 --dump-forces-fmt=%8.4f"
MPIPROCS 6
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=6
type=driver
# this is to test a different name
arg="--plumed plumed.dat --mf_xtc traj.xtc --dump-forces forces --multi 6 --dump-forces-fmt=%8.4f"
]] )
