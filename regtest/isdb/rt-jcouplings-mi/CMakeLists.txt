PLUMED_TEST(rt-jcouplings-mi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 2 --timestep 0.002 --mf_xtc trajectory.xtc --dump-forces atom_forces --dump-forces-fmt=%7.3f"
NEEDS cregex
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs="cregex"
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 2 --timestep 0.002 --mf_xtc trajectory.xtc --dump-forces atom_forces --dump-forces-fmt=%7.3f"
]] )
