PLUMED_TEST(rt-test TYPE driver
ARGS "--plumed plumed.dat --ixyz diala_traj_nm.xyz"
MODULES annfunc
EXTRAFILES ../../trajectories/diala_traj_nm.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=annfunc
arg="--plumed plumed.dat --ixyz diala_traj_nm.xyz"
extra_files="../../trajectories/diala_traj_nm.xyz"
]] )
