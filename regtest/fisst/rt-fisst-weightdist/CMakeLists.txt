PLUMED_TEST(rt-fisst-weightdist TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz --kt 0.593"
MODULES fisst
EXTRAFILES ../../trajectories/trajectory.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=fisst
arg="--plumed plumed.dat --ixyz trajectory.xyz --kt 0.593"
extra_files="../../trajectories/trajectory.xyz"

]] )
