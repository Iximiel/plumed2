PLUMED_TEST(rt-opes_expanded-init TYPE driver
ARGS "--plumed plumed.dat --mf_xtc chignolin.xtc --dump-forces forces --dump-forces-fmt=%8.4f"
MODULES opes
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=opes
type=driver
arg="--plumed plumed.dat --mf_xtc chignolin.xtc --dump-forces forces --dump-forces-fmt=%8.4f"
]] )
