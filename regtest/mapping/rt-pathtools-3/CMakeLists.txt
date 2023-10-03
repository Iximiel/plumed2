PLUMED_TEST(rt-pathtools-3 TYPE plumed
ARGS "pathtools --path all.pdb --fixed 3,6 --metric OPTIMAL --out path-out.pdb"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
arg="pathtools --path all.pdb --fixed 3,6 --metric OPTIMAL --out path-out.pdb" 
]] )
