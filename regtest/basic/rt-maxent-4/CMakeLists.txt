PLUMED_TEST(rt-maxent-4 TYPE driver
ARGS "--plumed plumed.dat --mf_xtc traj.xtc --dump-forces forces --dump-forces-fmt=%8.2f --debug-dd yes"
MPIPROCS 6
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=6
type=driver
arg="--plumed plumed.dat --mf_xtc traj.xtc --dump-forces forces --dump-forces-fmt=%8.2f --debug-dd yes"
]] )
