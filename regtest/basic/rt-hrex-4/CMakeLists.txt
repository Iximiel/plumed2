PLUMED_TEST(rt-hrex-4 TYPE plumed
ARGS "partial_tempering 0.5 < processed.top"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
# this is to test a different name
arg="partial_tempering 0.5 < processed.top"
]] )
