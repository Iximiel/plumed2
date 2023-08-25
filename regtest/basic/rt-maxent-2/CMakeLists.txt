PLUMED_TEST(rt-maxent-2 TYPE driver
ARGS "--plumed plumed.dat --mf_xtc traj.xtc --dump-forces forces --dump-forces-fmt=%8.2f"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --mf_xtc traj.xtc --dump-forces forces --dump-forces-fmt=%8.2f"
]] )
