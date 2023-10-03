PLUMED_TEST(rt-pathtools-2 TYPE plumed
ARGS "pathtools --start start.pdb --end end.pdb --nframes 1 --metric OPTIMAL --out mypath.pdb"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
arg="pathtools --start start.pdb --end end.pdb --nframes 1 --metric OPTIMAL --out mypath.pdb" 
]] )
