PLUMED_TEST(NAME rt-molinfo TYPE driver
ARGS "--igro test.gro --plumed plumed.dat"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--igro test.gro --plumed plumed.dat"
]] )
