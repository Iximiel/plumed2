PLUMED_TEST(rt-saxs-2 TYPE driver
ARGS "--plumed plumed.dat --mf_pdb template.pdb"
MPIPROCS 2
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
arg="--plumed plumed.dat --mf_pdb template.pdb"
]] )
