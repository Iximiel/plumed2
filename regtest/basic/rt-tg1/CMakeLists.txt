PLUMED_TEST(NAME rt-tg1 TYPE driver
ARGS "--plumed plumed.dat --length-units A --ixyz tip3.A.xyz --pdb tip3.plumed.pdb"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --length-units A --ixyz tip3.A.xyz --pdb tip3.plumed.pdb"
]] )
