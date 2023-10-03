PLUMED_TEST(rt-vimsyntax TYPE plumed
ARGS "vim2html --annotate-syntax plumed.dat annotation.dat"
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
# this is to test a different name
arg="vim2html --annotate-syntax plumed.dat annotation.dat"
]] )
