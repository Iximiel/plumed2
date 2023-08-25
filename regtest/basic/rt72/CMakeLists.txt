PLUMED_TEST(rt72 TYPE driver
ARGS "--plumed plumed.dat --igro trpcage.gro"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --igro trpcage.gro"
]] )
