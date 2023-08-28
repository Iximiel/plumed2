PLUMED_TEST(rt-pathtools-1 TYPE plumed
ARGS "pathtools --start start.pdb --end end.pdb --nframes 5 --nframes-after-end 4 --metric EUCLIDEAN --out mypath.pdb --arg-fmt %8.4f"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
arg="pathtools --start start.pdb --end end.pdb --nframes 5 --nframes-after-end 4 --metric EUCLIDEAN --out mypath.pdb --arg-fmt %8.4f" 
]] )
