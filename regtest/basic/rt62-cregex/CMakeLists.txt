PLUMED_TEST(rt62-cregex TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 1 --ixyz diala_traj_nm.xyz  "
NEEDS cregex
EXTRAFILES ../../trajectories/diala_traj_nm.xyz
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs="cregex"
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 1 --ixyz diala_traj_nm.xyz  "
extra_files="../../trajectories/diala_traj_nm.xyz"
]] )
