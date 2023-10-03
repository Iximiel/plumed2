PLUMED_TEST(rt-saxs TYPE driver
ARGS "--plumed plumed.dat --timestep 0.005 --mf_pdb template.pdb"
MPIPROCS 4
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
arg="--plumed plumed.dat --timestep 0.005 --mf_pdb template.pdb"
]] )
