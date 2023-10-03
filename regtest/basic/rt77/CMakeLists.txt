PLUMED_TEST(rt77 TYPE driver
ARGS "--plumed plumed.dat --timestep 0.002 --mf_xtc trajectory.xtc --dump-forces forces --dump-forces-fmt=%.2f"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --timestep 0.002 --mf_xtc trajectory.xtc --dump-forces forces --dump-forces-fmt=%.2f"
]] )
