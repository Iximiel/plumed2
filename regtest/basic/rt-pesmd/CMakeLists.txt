PLUMED_TEST(rt-pesmd TYPE plumed
ARGS "pesmd input.dat"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
arg="pesmd input.dat" 
]] )
