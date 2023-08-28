PLUMED_TEST(rt-divergence TYPE plumed
ARGS "drr_tool --divergence eabf2dpbc.drrstate"
MODULES drr
NEEDS boost_serialization
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
plumed_needs=boost_serialization
plumed_modules=drr
arg="drr_tool --divergence eabf2dpbc.drrstate"
]] )
