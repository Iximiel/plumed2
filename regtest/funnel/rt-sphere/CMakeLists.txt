PLUMED_TEST(rt-sphere TYPE driver
ARGS "--plumed plumed.dat --mf_trr small.trr --pdb start.pdb --dump-forces forces --dump-forces-fmt=%10.4f"
MPIPROCS 1
MODULES funnel
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=funnel
mpiprocs=1
type=driver
arg="--plumed plumed.dat --mf_trr small.trr --pdb start.pdb --dump-forces forces --dump-forces-fmt=%10.4f"
]] )
