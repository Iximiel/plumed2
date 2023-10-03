PLUMED_TEST(rt-close-structure TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 50 --timestep 0.005 --ixyz diala_traj_nm.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
EXTRAFILES ../../trajectories/diala_traj_nm.xyz  ../../trajectories/path_msd/allv.pdb 
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 50 --timestep 0.005 --ixyz diala_traj_nm.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
extra_files="../../trajectories/diala_traj_nm.xyz  ../../trajectories/path_msd/allv.pdb "
]] )
