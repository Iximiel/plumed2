PLUMED_TEST(rt-md_le-2d-wolfe-quapp-rotated TYPE plumed
ARGS "ves_md_linearexpansion input"
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
type=plumed
arg="ves_md_linearexpansion input"

]] )
