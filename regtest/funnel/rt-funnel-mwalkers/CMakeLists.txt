PLUMED_TEST(rt-funnel-mwalkers TYPE driver
ARGS "--plumed plumed.dat --mf_trr small.trr --pdb start.pdb --multi 2 --dump-forces forces --dump-forces-fmt=%10.4f"
MPIPROCS 2
MODULES funnel
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=funnel
mpiprocs=2
type=driver
arg="--plumed plumed.dat --mf_trr small.trr --pdb start.pdb --multi 2 --dump-forces forces --dump-forces-fmt=%10.4f"
]] )
