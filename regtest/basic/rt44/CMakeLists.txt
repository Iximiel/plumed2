PLUMED_TEST(NAME rt44 TYPE sum_hills
ARGS " --min -pi,pi --max -pi,pi --bin 99,99 --hills HILLS_t1 --fmt %12.7f"
EXTRAFILES ../../trajectories/HILLS_t1
)
#the following variable is the original config file
set(originalConfig [[
type=sum_hills
# this is to test a different name
arg=" --min -pi,pi --max -pi,pi --bin 99,99 --hills HILLS_t1 --fmt %12.7f"
extra_files="../../trajectories/HILLS_t1"
]] )
