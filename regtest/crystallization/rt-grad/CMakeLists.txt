PLUMED_TEST(rt-grad TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz LJ_VAP.xyz --box 7.73950,7.73950,7.73950 --dump-forces forces --dump-forces-fmt=%8.4f"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
# this is to test a different name
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz LJ_VAP.xyz --box 7.73950,7.73950,7.73950 --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
