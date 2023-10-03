PLUMED_TEST(rt-ermsd2 TYPE driver
ARGS "--plumed plumed.dat --mf_xtc traj.xtc"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
#arg="--plumed plumed.dat --mf_xtc samples.xtc"
#arg="--plumed plumed.dat --igro broken.gro"
#arg="--plumed plumed.dat --igro before.gro"
arg="--plumed plumed.dat --mf_xtc traj.xtc"
]] )
