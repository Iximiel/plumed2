PLUMED_TEST(rt-emmi-out TYPE driver
ARGS "--plumed plumed.dat --mf_pdb structure.pdb --box 100,100,100"
MPIPROCS 1
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=1
type=driver
arg="--plumed plumed.dat --mf_pdb structure.pdb --box 100,100,100" 
]] )
