PLUMED_TEST(rt33 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz amyloid.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz amyloid.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
]] )
