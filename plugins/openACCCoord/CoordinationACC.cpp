/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2013-2023 The plumed team
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
#include "plumed/core/Colvar.h"
#include "plumed/core/ActionRegister.h"

#include "plumed/tools/SwitchingFunction.h"
#include "plumed/tools/NeighborList.h"

// #include "plumed/tools/Communicator.h"
#include "fastCoord.hxx"

#include <string_view>
#include <iostream>
#define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'

namespace PLMD {
namespace colvar {


class CoordinationACC : public Colvar {
  bool pbc;
  bool serial;
  std::unique_ptr<NeighborList> nl;
  bool invalidateList;
  bool firsttime;

  unsigned natA{0};
  unsigned natB{0};
  //used to get some calculations
  SwitchingFunction switching;
public:
  explicit CoordinationACC(const ActionOptions&);
  ~CoordinationACC();
// active methods:
  void calculate() override;
  void prepare() override;
  double pairing(double distance,double&dfunc,unsigned i,unsigned j);
  static void registerKeywords( Keywords& keys );
};
void CoordinationACC::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("PAIR",false,"Pair only 1st element of the 1st group with 1st element in the second, etc");
  keys.addFlag("NLIST",false,"Use a neighbor list to speed up the calculation");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbor list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbor list");
  keys.add("atoms","GROUPA","First list of atoms");
  keys.add("atoms","GROUPB","Second list of atoms (if empty, N*(N-1)/2 pairs in GROUPA are counted)");
}

PLUMED_REGISTER_ACTION(CoordinationACC,"COORDINATIONACC")


CoordinationACC::CoordinationACC(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  serial(false),
  invalidateList(true),
  firsttime(true) {

  parseFlag("SERIAL",serial);

  switching.set(6,0,1.0,0.0);

  std::vector<AtomNumber> ga_lista,gb_lista;
  parseAtomList("GROUPA",ga_lista);
  natA=ga_lista.size();
  parseAtomList("GROUPB",gb_lista);
  natB=gb_lista.size();
  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

// pair stuff
  bool dopair=false;
  parseFlag("PAIR",dopair);

// neighbor list stuff
  bool doneigh=false;
  double nl_cut=0.0;
  int nl_st=0;
  parseFlag("NLIST",doneigh);
  if(doneigh) {
    parse("NL_CUTOFF",nl_cut);
    if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
    parse("NL_STRIDE",nl_st);
    if(nl_st<=0) error("NL_STRIDE should be explicitly specified and positive");
  }

  addValueWithDerivatives(); setNotPeriodic();
  if(gb_lista.size()>0) {
    if(doneigh)  nl=Tools::make_unique<NeighborList>(ga_lista,gb_lista,serial,dopair,pbc,getPbc(),comm,nl_cut,nl_st);
    else         nl=Tools::make_unique<NeighborList>(ga_lista,gb_lista,serial,dopair,pbc,getPbc(),comm);
  } else {
    if(doneigh)  nl=Tools::make_unique<NeighborList>(ga_lista,serial,pbc,getPbc(),comm,nl_cut,nl_st);
    else         nl=Tools::make_unique<NeighborList>(ga_lista,serial,pbc,getPbc(),comm);
  }

  requestAtoms(nl->getFullAtomList());

  log.printf("  between two groups of %u and %u atoms\n",static_cast<unsigned>(ga_lista.size()),static_cast<unsigned>(gb_lista.size()));
  log.printf("  first group:\n");
  for(unsigned int i=0; i<ga_lista.size(); ++i) {
    if ( (i+1) % 25 == 0 ) log.printf("  \n");
    log.printf("  %d", ga_lista[i].serial());
  }
  log.printf("  \n  second group:\n");
  for(unsigned int i=0; i<gb_lista.size(); ++i) {
    if ( (i+1) % 25 == 0 ) log.printf("  \n");
    log.printf("  %d", gb_lista[i].serial());
  }
  log.printf("  \n");
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");
  if(dopair) log.printf("  with PAIR option\n");
  if(doneigh) {
    log.printf("  using neighbor lists with\n");
    log.printf("  update every %d steps and cutoff %f\n",nl_st,nl_cut);
  }
}

CoordinationACC::~CoordinationACC() {
// destructor required to delete forward declared class
}

void CoordinationACC::prepare() {
  if(nl->getStride()>0) {
    if(firsttime || (getStep()%nl->getStride()==0)) {
      requestAtoms(nl->getFullAtomList());
      invalidateList=true;
      firsttime=false;
    } else {
      requestAtoms(nl->getReducedAtomList());
      invalidateList=false;
      if(getExchangeStep()) error("Neighbor lists should be updated on exchange steps - choose a NL_STRIDE which divides the exchange stride!");
    }
    if(getExchangeStep()) firsttime=true;
  }
}

double pairing(double distance,double&dfunc,unsigned i,unsigned j) {
  return 0.0;
}

// calculator
void CoordinationACC::calculate()
{

  double ncoord=0.;
  Tensor virial;
  std::vector<Vector> deriv(getNumberOfAtoms());
  const float dmax=switching.get_dmax();
  const float invr02=[&]() {
    auto t= 1.0/switching.get_r0();
    return t*t;
  }();
  {

    std::vector<float> positions(3*getPositions().size());
    for(auto i=0U; i<getPositions().size(); ++i) {
      auto &tmp = getPosition(i);
      positions[i*3  ] = tmp[0];
      positions[i*3+1] = tmp[1];
      positions[i*3+2] = tmp[2];
    }
    std::vector<float> derivatives(3*getPositions().size());
    vdbg(derivatives.size());
    std::vector<float> virialF(9);


    ncoord = myACC::calculateSwitch(natA,natB,
                                    positions.data(),derivatives.data(),virialF.data(),invr02,dmax);
    for(auto i=0U; i<getPositions().size(); ++i) {
      deriv[i][0]=derivatives[i*3  ];
      deriv[i][1]=derivatives[i*3+1];
      deriv[i][2]=derivatives[i*3+2];
    }
    constexpr int dbgnum=1;
    vdbg(derivatives[3*dbgnum+0]);
    vdbg(derivatives[3*dbgnum+1]);
    vdbg(derivatives[3*dbgnum+2]);
    vdbg(deriv[dbgnum]);
    for(auto i=0U,k=0U; i<3U; ++i) {
      for(auto j=0U; j<3U; ++j) {
        virial[i][j]=virialF[k];
        ++k;
      }
    }
  }

  for(unsigned i=0; i<deriv.size(); ++i) {
    setAtomsDerivatives(i,deriv[i]);
  }
  setValue           (ncoord);
  setBoxDerivatives  (virial);

}
} // namespace colvar
}//namespace PLMD
