PLUMED_TEST(NAME rt-make-threads TYPE make
NEEDS molfile_plugins
)
#the following variable is the original config file
set(originalConfig [[
type=make
plumed_needs=molfile_plugins
]] )
