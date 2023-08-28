PLUMED_TEST(rt-emmi-gauss-mpi TYPE driver
ARGS "--plumed plumed.dat --mf_pdb structure.pdb --box 100,100,100"
MPIPROCS 2
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
arg="--plumed plumed.dat --mf_pdb structure.pdb --box 100,100,100" 
]] )
