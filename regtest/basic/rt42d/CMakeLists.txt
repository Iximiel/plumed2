PLUMED_TEST(NAME rt42d TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz --dump-forces ff --dump-forces-fmt=%10.5f --dump-full-virial"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --ixyz trajectory.xyz --dump-forces ff --dump-forces-fmt=%10.5f --dump-full-virial"
]] )
