PLUMED_TEST(rt-molfile-3 TYPE driver
ARGS "--plumed plumed.dat --mf_xtc test.xtc"
NEEDS molfile_plugins
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_needs=molfile_plugins
arg="--plumed plumed.dat --mf_xtc test.xtc"
]] )
