enable_testing()

set(type "driver")
set (plmdarg --plumed plumed.dat --ixyz trajectory.xyz --dump-forces ff --dump-forces-fmt=%10.5f --dump-full-virial)

PLUMEDTEST(rt42 "driver --plumed plumed.dat --ixyz trajectory.xyz --dump-forces ff --dump-forces-fmt=%10.5f --dump-full-virial")
