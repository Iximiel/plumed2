PLUMED_TEST(rt-opes_expanded TYPE driver
ARGS "--plumed plumed.dat --mf_xtc alanine.xtc --dump-forces forces --dump-forces-fmt=%8.4f"
MPIPROCS 2
MODULES opes
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=opes
mpiprocs=2
type=driver
arg="--plumed plumed.dat --mf_xtc alanine.xtc --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
