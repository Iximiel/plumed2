PLUMED_TEST(rt-extract2dpbc TYPE plumed
ARGS "drr_tool --extract eabf2dpbc.drrstate"
MODULES drr
NEEDS boost_serialization
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
plumed_needs=boost_serialization
plumed_modules=drr
arg="drr_tool --extract eabf2dpbc.drrstate"
]] )
