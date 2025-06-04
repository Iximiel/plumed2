/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2013-2020 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifdef __PLUMED_HAS_OPENACC
#define __PLUMED_USE_OPENACC 1
#endif //__PLUMED_HAS_OPENACC
#include "core/ActionRegister.h"
#include "tools/Pbc.h"
#include "tools/HistogramBead.h"
#include "ActionVolume.h"
#include "VolumeShortcut.h"

//+PLUMEDOC VOLUMES AROUND
/*
This quantity can be used to calculate functions of the distribution of collective variables for the atoms that lie in a particular, user-specified part of of the cell.

This action can be used to calculate whether each of the atoms are within a particular part of the simulation box or not as illustrated by the following example:

```plumed
f: FIXEDATOM AT=0,0,0
a: AROUND ATOMS=1-100 ORIGIN=f SIGMA=0.2 XLOWER=-1.0 XUPPER=1.0 YLOWER=-1.0 YUPPER=1.0 ZLOWER=-1.0 ZUPPER=1.0
PRINT ARG=a FILE=colvar
```

The 100 elements of the vector `a` that is returned from the AROUND action in the above input are calculated using:

$$
w(x_i,y_i,z_i) = \int_{xl}^{xu} \int_{yl}^{yu} \int_{zl}^{zu} \textrm{d}x\textrm{d}y\textrm{d}z K\left( \frac{x - x_i}{\sigma} \right)K\left( \frac{y - y_i}{\sigma} \right)K\left( \frac{z - z_i}{\sigma} \right)
$$

where $K$ is one of the kernel functions described in the documentation for the function [BETWEEN](BETWEEN.md), $\sigma$ is a bandwidth parameter and the limits
for the integrals are the values specified using the keywords XLOWER, XUPPER, YLOWER, YUPPER, YUPPER, ZLOWER and ZUPPER.  $x_i$, $y_i$ and $z_i$ are then the components
of the vector that connects the $i$th atom that was specified using the ATOMS keyword to the atom that was specified using the ORIGIN keyword.  In other words,
$w(x_i,y_i,z_i)$ is 1 if the atom is within a rectangular box that is centered on the atom that is specified as the origin and zero otherwise.

## Calculating the number of atoms in a particular part of the box

Lets suppose that you want to calculate how many atoms are in have an $x$ coordinate that is between -1.0 and 1.0. You can do this using the following PLUMED input:

```plumed
f: FIXEDATOM AT=0,0,0
a: AROUND ATOMS=1-100 ORIGIN=f SIGMA=0.2 XLOWER=-1.0 XUPPER=1.0
s: SUM ARG=a PERIODIC=NO
PRINT ARG=s FILE=colvar
```

In this example the components of `a` are calculated as:

$$
w(x_i,y_i,z_i) = \int_{xl}^{xu} \textrm{d}x K\left( \frac{x - x_i}{\sigma} \right)
$$

as the YLOWER, YUPPER, YUPPER, ZLOWER and ZUPPER flags have not been included.  The [SUM](SUM.md) command then adds together all the elements of the vector `a` to calculate the total number of atoms in the region
of the box that is of interest.

## Calculating the average value for an order parameter in a particular part of the box

Suppose that you have calculated a vector of order parameters that can be assigned to a particular point in three dimensional space.
The symmetry functions in the [symfunc](module_symfunc.md) module are examples of order parameters that satisfy this criteria. You can use
the AROUND command to calculate the average value of the symmetry function in a particular part of the box as follows:

```plumed
c: COORDINATIONNUMBER SPECIES=1-100 SWITCH={RATIONAL R_0=1.0}
f: FIXEDATOM AT=0,0,0
a: AROUND ATOMS=1-100 ORIGIN=f SIGMA=0.2 XLOWER=-1.0 XUPPER=1.0 YLOWER=-1.0 YUPPER=1.0
p: CUSTOM ARG=c,a FUNC=x*y PERIODIC=NO
n: SUM ARG=p PERIODIC=NO
d: SUM ARG=a PERIODIC=NO
av: CUSTOM ARG=n,d FUNC=x/y PERIODIC=NO
PRINT ARG=av FILE=colvar
```

The final quantity `av` here is:

$$
\overline{s}_{\tau} = \frac{ \sum_i c_i w(x_i,y_i,z_i) }{ \sum_i w(x_i,y_i,z_i) }
$$

where $c_i$ are the coordination numbers and $w_i$ is:

$$
w(x_i,y_i,z_i) = \int_{xl}^{xu} \int_{yl}^{yu} \textrm{d}x \textrm{d}y K\left( \frac{x - x_i}{\sigma} \right) K\left( \frac{y - y_i}{\sigma} \right)
$$

## Old syntax

In earlier versions of PLUMED the syntax for the calculation in the previous section is as follows:

```plumed
c: COORDINATIONNUMBER SPECIES=1-100 SWITCH={RATIONAL R_0=1.0}
f: FIXEDATOM AT=0,0,0
a: AROUND DATA=c ORIGIN=f SIGMA=0.2 XLOWER=-1.0 XUPPER=1.0 YLOWER=-1.0 YUPPER=1.0 MEAN
PRINT ARG=a.mean FILE=colvar
```

This old syntax still works but we highly recommend you use the newer syntax as it is easlier to understand
and more flexible. You will also notice that AROUND in the input above is a shortcut that expands to the longer input
that was given in the previous section.

*/
//+ENDPLUMEDOC

//+PLUMEDOC MCOLVAR AROUND_CALC
/*
Calculate a vector from the input positions with elements equal to one when the positions are in a particular part of the cell and elements equal to zero otherwise

\par Examples

*/
//+ENDPLUMEDOC

namespace PLMD {
namespace volumes {

class VolumeAround {
public:
  bool dox{true}, doy{true}, doz{true};
  double sigma;
  double xlow{0.0}, xhigh{0.0};
  double ylow{0.0}, yhigh{0.0};
  double zlow{0.0}, zhigh{0.0};
  HistogramBead::KernelType kerneltype;
  static void registerKeywords( Keywords& keys );
  void parseInput( ActionVolume<VolumeAround>* action );
  void setupRegions( ActionVolume<VolumeAround>* action,
                     const Pbc& pbc,
                     const std::vector<Vector>& positions ) {}
  static void parseAtoms( ActionVolume<VolumeAround>* action,
                          std::vector<AtomNumber>& atom );
  static void calculateNumberInside( const VolumeInput& input,
                                     const VolumeAround& actioninput,
                                     VolumeOutput& output );
#ifdef __PLUMED_USE_OPENACC
  void toACCDevice() const {
#pragma acc enter data copyin(this[0:1],dox,doy,doz,sigma,\
  xlow,xhigh,ylow,yhigh,zlow,zhigh,kerneltype)
  }
  void removeFromACCDevice() const {
#pragma acc exit data delete(kerneltype,zhigh,zlow,yhigh,ylow,xhigh,xlow,\
  sigma,doz,doy,dox,this[0:1])
  }
#endif //__PLUMED_USE_OPENACC
};

typedef ActionVolume<VolumeAround> Vola;
PLUMED_REGISTER_ACTION(Vola,"AROUND_CALC")
char glob_around[] = "AROUND";
typedef VolumeShortcut<glob_around> VolumeAroundShortcut;
PLUMED_REGISTER_ACTION(VolumeAroundShortcut,"AROUND")

void VolumeAround::registerKeywords( Keywords& keys ) {
  keys.setDisplayName("AROUND");
  keys.add("atoms","ORIGIN","the atom whose vicinity we are interested in examining");
  keys.add("atoms-2","ATOM","an alternative to ORIGIN");
  keys.add("compulsory","SIGMA","the width of the function to be used for kernel density estimation");
  keys.add("compulsory","KERNEL","gaussian","the type of kernel function to be used");
  keys.add("compulsory","XLOWER","0.0","the lower boundary in x relative to the x coordinate of the atom (0 indicates use full extent of box).");
  keys.add("compulsory","XUPPER","0.0","the upper boundary in x relative to the x coordinate of the atom (0 indicates use full extent of box).");
  keys.add("compulsory","YLOWER","0.0","the lower boundary in y relative to the y coordinate of the atom (0 indicates use full extent of box).");
  keys.add("compulsory","YUPPER","0.0","the upper boundary in y relative to the y coordinate of the atom (0 indicates use full extent of box).");
  keys.add("compulsory","ZLOWER","0.0","the lower boundary in z relative to the z coordinate of the atom (0 indicates use full extent of box).");
  keys.add("compulsory","ZUPPER","0.0","the upper boundary in z relative to the z coordinate of the atom (0 indicates use full extent of box).");
}

void VolumeAround::parseAtoms( ActionVolume<VolumeAround>* action, std::vector<AtomNumber>& atom ) {
  action->parseAtomList("ORIGIN",atom);
  if( atom.size()==0 ) {
    action->parseAtomList("ATOM",atom);
  }
  if( atom.size()!=1 ) {
    action->error("should only be one atom specified");
  }
  action->log.printf("  boundaries for region are calculated based on positions of atom : %d\n",atom[0].serial() );
}

void VolumeAround::parseInput( ActionVolume<VolumeAround>* action ) {
  action->parse("SIGMA",sigma);
  std::string mykerneltype;
  action->parse("KERNEL",mykerneltype);
  kerneltype=HistogramBead::getKernelType(mykerneltype);
  dox=true;
  action->parse("XLOWER",xlow);
  action->parse("XUPPER",xhigh);
  doy=true;
  action->parse("YLOWER",ylow);
  action->parse("YUPPER",yhigh);
  doz=true;
  action->parse("ZLOWER",zlow);
  action->parse("ZUPPER",zhigh);
  if( xlow==0.0 && xhigh==0.0 ) {
    dox=false;
  }
  if( ylow==0.0 && yhigh==0.0 ) {
    doy=false;
  }
  if( zlow==0.0 && zhigh==0.0 ) {
    doz=false;
  }
  if( !dox && !doy && !doz ) {
    action->error("no subregion defined use XLOWER, XUPPER, YLOWER, YUPPER, ZLOWER, ZUPPER");
  }
  action->log.printf("  boundaries for region (region of interest about atom) : x %f %f, y %f %f, z %f %f \n",xlow,xhigh,ylow,yhigh,zlow,zhigh);
}

void VolumeAround::calculateNumberInside( const VolumeInput& input,
    const VolumeAround& actioninput,
    VolumeOutput& output ) {
  // Setup the histogram bead
  HistogramBead bead;
  bead.isNotPeriodic();
  bead.setKernelType( actioninput.kerneltype );

  // Calculate position of atom wrt to origin
  Vector fpos=input.pbc.distance( Vector(input.refpos[0][0],input.refpos[0][1],input.refpos[0][2]),
                                  Vector(input.cpos[0],input.cpos[1],input.cpos[2]) );
  double xcontr, ycontr, zcontr, xder, yder, zder;
  if( actioninput.dox ) {
    bead.set( actioninput.xlow, actioninput.xhigh, actioninput.sigma );
    xcontr=bead.calculate( fpos[0], xder );
  } else {
    xcontr=1.;
    xder=0.;
  }
  if( actioninput.doy ) {
    bead.set( actioninput.ylow, actioninput.yhigh, actioninput.sigma );
    ycontr=bead.calculate( fpos[1], yder );
  } else {
    ycontr=1.;
    yder=0.;
  }
  if( actioninput.doz ) {
    bead.set( actioninput.zlow, actioninput.zhigh, actioninput.sigma );
    zcontr=bead.calculate( fpos[2], zder );
  } else {
    zcontr=1.;
    zder=0.;
  }
  output.derivatives[0]=xder*ycontr*zcontr;
  output.derivatives[1]=xcontr*yder*zcontr;
  output.derivatives[2]=xcontr*ycontr*zder;
  // Add derivatives wrt to position of origin atom
  output.refders[0][0] = -output.derivatives[0];
  output.refders[0][1] = -output.derivatives[1];
  output.refders[0][2] = -output.derivatives[2];
  // Add virial contribution
  output.virial.set( 0, -Tensor(fpos,
                                Vector(output.derivatives[0], output.derivatives[1], output.derivatives[2])) );
  output.values[0] = xcontr*ycontr*zcontr;
}

}
}
