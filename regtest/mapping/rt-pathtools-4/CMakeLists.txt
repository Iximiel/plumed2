PLUMED_TEST(rt-pathtools-4 TYPE plumed
ARGS "pathtools --path epath.pdb --metric EUCLIDEAN --out epath-out.pdb --arg-fmt %8.4f"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
arg="pathtools --path epath.pdb --metric EUCLIDEAN --out epath-out.pdb --arg-fmt %8.4f" 
]] )
