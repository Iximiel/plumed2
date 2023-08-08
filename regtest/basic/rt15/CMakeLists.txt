enable_testing()

set(type "driver")
set (plmdarg --plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz)
set(plumed_needs "dlopen")


PLUMED_TEST(NAME rt15
    TYPE driver
    ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz"
    NEEDS "dlopen")
