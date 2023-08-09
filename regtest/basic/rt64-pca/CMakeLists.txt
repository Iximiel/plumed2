PLUMED_TEST(NAME rt64-pca TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 1 --ixyz diala_traj_nm.xyz  "
EXTRAFILES ../../trajectories/diala_traj_nm.xyz ../../trajectories/pca/eigenvec.pdb ../../trajectories/pca/average.pdb ../../trajectories/pca/eigenvec1.pdb
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 1 --ixyz diala_traj_nm.xyz  "
extra_files="../../trajectories/diala_traj_nm.xyz ../../trajectories/pca/eigenvec.pdb ../../trajectories/pca/average.pdb ../../trajectories/pca/eigenvec1.pdb"
]] )
