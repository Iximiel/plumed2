PLUMED_TEST(rt-pca TYPE driver
ARGS "--plumed plumed.dat --ixyz diala_traj_nm.xyz"
MODULES dimred
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=dimred
arg="--plumed plumed.dat --ixyz diala_traj_nm.xyz"
]] )
