PLUMED_TEST(NAME rt71 TYPE driver
ARGS "--plumed plumed.dat --ixyz trajectory.xyz --dump-forces ff --dump-forces-fmt=%10.6f --dump-full-virial"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --ixyz trajectory.xyz --dump-forces ff --dump-forces-fmt=%10.6f --dump-full-virial"
]] )
