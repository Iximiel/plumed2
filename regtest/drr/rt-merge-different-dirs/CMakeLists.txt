PLUMED_TEST(rt-merge-different-dirs TYPE plumed
ARGS "drr_tool --merge ./dir1/eabf_win1.drrstate,./dir2/eabf_win2.drrstate --merge_output merged"
MODULES drr
NEEDS boost_serialization
)
#the following variable is the original config file
set(originalConfig [[
type=plumed
plumed_needs=boost_serialization
plumed_modules=drr
arg="drr_tool --merge ../dir1/eabf_win1.drrstate,../dir2/eabf_win2.drrstate --merge_output merged"
]] )
