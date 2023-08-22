PLUMED_TEST(NAME rt-hrex-2 TYPE plumed
ARGS "partial_tempering 0.5 < processed.top"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
# this is to test a different name
arg="partial_tempering 0.5 < processed.top"
]] )
