PLUMED_TEST(NAME rt-fix-226 TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz --length-units=A"
)
#the following variable is the original config file
set(originalConfig [[
type=driver
arg="--plumed plumed.dat --ixyz traj.xyz --length-units=A"

# this is to avoid storing a huge traj file
# just to have >100k atoms
function plumed_regtest_before(){
  awk 'BEGIN{
    n=100010;
    print n
    print 0,0,0
    for(i=1;i<=n;i++) print "Ar",i*0.001,i*0.001,i*0.001
  }' > traj.xyz
}

]] )
