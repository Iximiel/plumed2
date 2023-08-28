PLUMED_TEST(rt-cavity TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz rotating_box.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz rotating_box.xyz --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
