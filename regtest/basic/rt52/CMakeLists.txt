PLUMED_TEST(rt52 TYPE sum_hills
ARGS " --negbias --bin 30,30 --hills HILLS_t2 --fmt %8.3f "
EXTRAFILES ../../trajectories/HILLS_t2 
)
#the following variable is the original config file
set(originalConfig [[
type=sum_hills
# this is to test the welltempered negbias 
arg=" --negbias --bin 30,30 --hills HILLS_t2 --fmt %8.3f "
extra_files="../../trajectories/HILLS_t2 "
]] )
