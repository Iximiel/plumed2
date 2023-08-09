PLUMED_TEST(NAME rt44b TYPE sum_hills
ARGS " --min -pi,-pi --max pi,pi --bin 20,20 --hills HILLS --fmt %12.7f"
)
#the following variable is the original config file
set(originalConfig [[
type=sum_hills
arg=" --min -pi,-pi --max pi,pi --bin 20,20 --hills HILLS --fmt %12.7f"
]] )
