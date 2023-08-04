enable_testing()

set(type "driver")
set (plmdarg --plumed plumed.dat --trajectory-stride 50 --timestep 0.005 --ixyz broken.xyz --dump-forces forces --dump-forces-fmt=%10.6f)

PLUMED_TEST(NAME rt40 TYPE driver ARGS "--plumed plumed.dat --trajectory-stride 50 --timestep 0.005 --ixyz broken.xyz --dump-forces forces --dump-forces-fmt=%10.6f")
