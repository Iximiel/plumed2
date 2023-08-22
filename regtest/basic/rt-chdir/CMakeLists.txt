PLUMED_TEST(NAME rt-chdir TYPE make
)
#the following variable is the original config file
set(originalConfig [[
type=make

function plumed_regtest_before() {
mkdir aa
touch aa/xx
touch aa/yy
}

]] )
