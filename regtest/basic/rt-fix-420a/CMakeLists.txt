PLUMED_TEST(rt-fix-420a TYPE sum_hills
ARGS "--hills HILLS --fmt %10.6f --min 0 --max 10 --spacing 0.1"
)
#the following variable is the original config file
set(originalConfig [[
type=sum_hills
# this is to test multiple input files 
arg="--hills HILLS --fmt %10.6f --min 0 --max 10 --spacing 0.1"
]] )
