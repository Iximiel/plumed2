PLUMED_TEST(rt-ecv_custom TYPE driver
ARGS "--plumed plumed.dat --mf_xtc alanine.xtc --dump-forces forces --dump-forces-fmt=%8.4f"
MODULES opes
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=opes
type=driver
arg="--plumed plumed.dat --mf_xtc alanine.xtc --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
