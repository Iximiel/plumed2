/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2025 of Alexander Humeniuk.

   This file is part of the liquid_crystal plumed module.

   The liquid_crystal plumed module is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The liquid_crystal plumed module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "core/ActionShortcut.h"
#include "core/ActionRegister.h"
#include "multicolvar/MultiColvarShortcuts.h"

using namespace PLMD::multicolvar;

namespace PLMD {
namespace liquid_crystal {

//+PLUMEDOC LIQUID_CRYSTAL NEMATIC_ORDER
/*
Calculate the nematic order parameter.

The nematic order parameter S characterizes the orientational order of molecules
and ranges from S=0 (isotropic) to S=1 (all molecular axes are perfectly parallel).
Most liquids are isotropic, as there is no preferred direction, and have an order parameter
close to 0. In liquid crystals, membranes and solids, molecules tend to align giving
rise to order parameters closer to 1.

$S$ is calculated from the distribution of the angles between the molecular axes ($\hat{u}_i$ for $i=1,\ldots,N$)
and the nematic director $\hat{n}$,
$$
S = \frac{1}{N} \sum_{i=1}^N \left(\frac{3}{2} \cos^2(\theta_i) - \frac{1}{2} \right),
$$
with $\cos(\theta_i) = \hat{n} \cdot \hat{u}_i$.

The nematic director depends on the distribution of the molecular axes
and is computed as the eigenvector belonging to the largest eigenvalue
of the $3 \times 3$ nematic order tensor,
$$
Q_{a,b} = \frac{1}{N} \sum_{i=1}^N \left(\frac{3}{2} u_{a,i} u_{b,i} - \frac{1}{2} \delta_{a,b} \right).
$$

By adding a bias to the nematic order parameter, one can drive a liquid crystal from the
isotropic to the nematic phase.

The axis of a rod-like molecule is defined as the distance vector between two atoms,
it points from the tail atom to the head atom.

```plumed
# Assume there are three molecules with 20 atoms each.
# In the first molecule the molecular axis vector points from atom 1 to atom 20,
# in the second molecule it points from atom 21 to atom 40
# and in the third from atom 41 to atom 60.

# Compute nematic order parameter for the three molecules.
S: NEMATIC_ORDER MOLECULE_STARTS=1,21,41 MOLECULE_ENDS=20,40,60
PRINT FILE=colvar ARG=S

# Add a bias to the nematic order parameter S.
BIASVALUE ARG=S
```

*/
//+ENDPLUMEDOC

class NematicOrder : public ActionShortcut {
  public:
    static void registerKeywords(Keywords& keys);
    explicit NematicOrder(const ActionOptions&);
  };

  PLUMED_REGISTER_ACTION(NematicOrder,"NEMATIC_ORDER")

  void NematicOrder::registerKeywords(Keywords& keys) {
    ActionShortcut::registerKeywords( keys );
    keys.add("atoms","MOLECULE_STARTS","The atoms where the molecular axis starts.");
    keys.add("atoms","MOLECULE_ENDS","The atoms where the molecular axis ends.");
    keys.setValueDescription("scalar","the modulus of the average vector");
    keys.needsAction("CONSTANT");
    keys.needsAction("CUSTOM");
    keys.needsAction("DIAGONALIZE");
    keys.needsAction("DISTANCE");
    keys.needsAction("MATRIX_PRODUCT");
    keys.needsAction("TRANSPOSE");
    keys.needsAction("VSTACK");
  }

  NematicOrder:: NematicOrder(const ActionOptions& ao):
    Action(ao),
    ActionShortcut(ao) {
      // Fetch indices of atoms that define the tails and the heads of the molecular axes.
      std::vector<std::string> starts, ends;
      MultiColvarShortcuts::parseAtomList("MOLECULE_STARTS",starts,this);
      MultiColvarShortcuts::parseAtomList("MOLECULE_ENDS",ends,this);

    if( starts.size()!=ends.size() ) error(
      "Mismatched numbers of atoms specified to MOLECULE_STARTS and MOLECULE_ENDS keywords. "
      "The molecular axes are specified by pairs of atoms."
    );

    std::string dlist = "";
    for(unsigned i=0; i<starts.size(); ++i) {
      std::string num;
      Tools::convert( i+1, num );
      dlist += " ATOMS" + num + "=" + starts[i] + "," + ends[i];
    }

    std::string L = getShortcutLabel();
    // Calculate the lengths of the distance vectors
    //   d: DISTANCE ATOMS1=1,2 ATOMS2=3,4 ATOMS3=5,6 ATOMS4=7,8
    readInputLine( L + "_dvals: DISTANCE" + dlist );
    // Calculate the vectorial orientations of the molecules
    //   dc: DISTANCE COMPONENTS ATOMS1=1,2 ATOMS2=3,4  ATOMS3=5,6 ATOMS4=7,8
    readInputLine( L + "_dvecs: DISTANCE COMPONENTS " + dlist );
    // Convert the vectors into unit vectors
    //   dux: CUSTOM ARG=dc.x,d FUNC=x/y PERIODIC=NO
    //   duy: CUSTOM ARG=dc.y,d FUNC=x/y PERIODIC=NO
    //   duz: CUSTOM ARG=dc.z,d FUNC=x/y PERIODIC=NO
    readInputLine( L + "_dux: CUSTOM ARG=" + L + "_dvecs.x," + L + "_dvals FUNC=x/y PERIODIC=NO");
    readInputLine( L + "_duy: CUSTOM ARG=" + L + "_dvecs.y," + L + "_dvals FUNC=x/y PERIODIC=NO");
    readInputLine( L + "_duz: CUSTOM ARG=" + L + "_dvecs.z," + L + "_dvals FUNC=x/y PERIODIC=NO");
    // Create a Nx3 matrix that contains all the unit vectors
    //   v: VSTACK ARG=dux,duy,duz
    readInputLine( L + "_v: VSTACK ARG=" + L + "_dux," + L + "_duy," + L + "_duz");
    //   vT: TRANSPOSE ARG=v 
    readInputLine( L + "_vT: TRANSPOSE ARG=" + L + "_v");
    // Now compute the 3x3 matrix Q
    //   c: CONSTANT VALUES=4,0,0,0,4,0,0,0,4 NROWS=3 NCOLS=3
    readInputLine( L + "_c: CONSTANT VALUES=4,0,0,0,4,0,0,0,4 NROWS=3 NCOLS=3");
    //   mp: MATRIX_PRODUCT ARG=vT,v
    readInputLine( L + "_mp: MATRIX_PRODUCT ARG=" + L + "_vT," + L + "_v");
    //   Q: CUSTOM ARG=mp,c FUNC=((3*x-y)/2)/4 PERIODIC=NO
    readInputLine( L + "_Q: CUSTOM ARG=" + L + "_mp," + L + "_c " + "FUNC=((3*x-y)/2)/4 PERIODIC=NO");
    // Diagonalize Q
    //   diag: DIAGONALIZE ARG=q
    readInputLine( L + "_diag: DIAGONALIZE ARG=" + L + "_Q");
    // Nematic order parameter is the largest eigenvalue of Q
    readInputLine( L + ": CUSTOM ARG=" + L + "_diag.vals-1 FUNC=x PERIODIC=NO");
  }

}
}
