PLUMED_TEST(rt-opes_metad_explore-mw-restart TYPE driver
ARGS "--plumed plumed.dat --mf_xtc alanine.xtc --dump-forces forces --dump-forces-fmt=%8.4f --multi 2"
MPIPROCS 2
MODULES opes
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=opes
mpiprocs=2
type=driver
arg="--plumed plumed.dat --mf_xtc alanine.xtc --dump-forces forces --dump-forces-fmt=%8.4f --multi 2"
]] )
