PLUMED_TEST(rt-protein TYPE python
ARGS "./python-script.py"
NEEDS python
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs=python
# this is to test a different name
type=python
arg="./python-script.py"
extra_files="../../trajectories/trajectory.xyz"
]] )
