PLUMED_TEST(rt-fps TYPE driver
ARGS "--plumed plumed.dat --mf_trr small.trr --pdb start.pdb"
MPIPROCS 1
MODULES funnel
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=1
type=driver
plumed_modules=funnel
arg="--plumed plumed.dat --mf_trr small.trr --pdb start.pdb"

]] )
