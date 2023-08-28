PLUMED_TEST(rt-sasa-hasel TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz ala12_trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
MODULES sasa
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=sasa
arg="--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz ala12_trajectory.xyz --dump-forces forces --dump-forces-fmt=%10.6f"
]] )
