PLUMED_TEST(rt-merge TYPE plumed
ARGS "drr_tool --merge eabf_win1.drrstate,eabf_win2.drrstate"
MODULES drr
NEEDS boost_serialization
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
plumed_needs=boost_serialization
plumed_modules=drr
arg="drr_tool --merge eabf_win1.drrstate,eabf_win2.drrstate"
]] )
