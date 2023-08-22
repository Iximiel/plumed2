PLUMED_TEST(NAME rt-fix-371c TYPE plumed
ARGS "pdbrenumber --ipdb in.pdb --opdb out.pdb --atomnumbers numbers.dat"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
arg="pdbrenumber --ipdb in.pdb --opdb out.pdb --atomnumbers numbers.dat"
]] )
