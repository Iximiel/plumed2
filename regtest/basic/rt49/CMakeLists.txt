PLUMED_TEST(NAME rt49 TYPE sum_hills
ARGS " --hills HILLS_t1  --fmt %8.3f "
EXTRAFILES ../../trajectories/HILLS_t1 
)
#the following variable is the original config file
set(originalConfig [[
type=sum_hills
# this is to test automatic sizing  
arg=" --hills HILLS_t1  --fmt %8.3f "
extra_files="../../trajectories/HILLS_t1 "
]] )
