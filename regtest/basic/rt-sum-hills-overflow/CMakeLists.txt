PLUMED_TEST(rt-sum-hills-overflow TYPE sum_hills
ARGS " --hills HILLS --fmt %12.7f --idw d1 --kt 0.1"
)
#the following variable is the original config file
set(originalConfig [[
type=sum_hills
# this is to test a different name
arg=" --hills HILLS --fmt %12.7f --idw d1 --kt 0.1"
]] )
