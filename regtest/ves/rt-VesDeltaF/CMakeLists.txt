PLUMED_TEST(rt-VesDeltaF TYPE driver
ARGS "--plumed plumed.dat --mf_xtc alanine.xtc --dump-forces forces --dump-forces-fmt=%8.4f"
MPIPROCS 2
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
mpiprocs=2
type=driver
arg="--plumed plumed.dat --mf_xtc alanine.xtc --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
