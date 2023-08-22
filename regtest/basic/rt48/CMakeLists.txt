PLUMED_TEST(NAME rt48 TYPE sum_hills
ARGS " --min -pi,pi --max -pi,pi --bin 49,49 --hills HILLS_t1a,HILLS_t1b  --fmt %8.3f "
EXTRAFILES ../../trajectories/HILLS_t1a ../../trajectories/HILLS_t1b
)
#the following variable is the original config file
set(originalConfig [[
type=sum_hills
# this is to test multiple input files 
arg=" --min -pi,pi --max -pi,pi --bin 49,49 --hills HILLS_t1a,HILLS_t1b  --fmt %8.3f "
extra_files="../../trajectories/HILLS_t1a ../../trajectories/HILLS_t1b"
]] )
