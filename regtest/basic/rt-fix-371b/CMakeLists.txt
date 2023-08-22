PLUMED_TEST(NAME rt-fix-371b TYPE plumed
ARGS "pdbrenumber --ipdb in.pdb --opdb out.pdb --firstatomnumber 99999"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
arg="pdbrenumber --ipdb in.pdb --opdb out.pdb --firstatomnumber 99999"
]] )
