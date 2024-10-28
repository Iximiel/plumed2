/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2013-2024 The plumed team
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
// #include "fastCoord.hxx"
#include "inlineSwitch.h"
#include "inlineAcc.h"

#include <string_view>
#include <iostream>
#include <numeric>
#include <variant>
#define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'

namespace PLMD {
namespace colvar {
template <typename T>
using mySWD= std::variant<::myACC::rationalData<T>,::myACC::switchData<T>>;

class CoordinationACC : public Colvar {

  bool pbc;
  bool serial;
  std::unique_ptr<NeighborList> nl;
  bool invalidateList;
  bool firsttime;

  //used to get some calculations
  mySWD<float> switchSettings;
public:
  explicit CoordinationACC(const ActionOptions&);
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
  keys.add("compulsory","NN","6","The n parameter of the switching function ");
  keys.add("compulsory","MM","0","The m parameter of the switching function; 0 implies 2*NN");
  keys.add("compulsory","D_0","0.0","The d_0 parameter of the switching function");
  keys.add("compulsory","R_0","The r_0 parameter of the switching function");
  // keys.add("optional","SWITCH","This keyword is used if you want to employ an alternative to the continuous switching function defined above. "
  //          "The following provides information on the \\ref switchingfunction that are available. "
  //          "When this keyword is present you no longer need the NN, MM, D_0 and R_0 keywords.");
}

PLUMED_REGISTER_ACTION(CoordinationACC,"COORDINATIONACC")


CoordinationACC::CoordinationACC(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  serial(false),
  invalidateList(true),
  firsttime(true) {
  parseFlag("SERIAL",serial);

  std::vector<AtomNumber> ga_lista,gb_lista;
  parseAtomList("GROUPA",ga_lista);
  unsigned natA=ga_lista.size();
  parseAtomList("GROUPB",gb_lista);
  unsigned natB=gb_lista.size();
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
  {
    // setting up the SwitchingFunction
    //using  PLMD::SwitchingFunction for configuration
    std::string sw;
    SwitchingFunction switchingFunction;
    unsigned nn=6;
    unsigned mm=0;
    // parse("SWITCH",sw);
    if(sw.length()>0) {
      std::string errors;
      switchingFunction.set(sw,errors);
      if( errors.length()!=0 ) {
        error("problem reading SWITCH keyword : " + errors );
      }
    } else {
      double d0=0.0;
      double r0=-1.0;
      parse("R_0",r0);
      if(r0<0.0) {
        error("R_0 should be explicitly specified and positive");
      }
      parse("D_0",d0);
      parse("NN",nn);
      parse("MM",mm);
      switchingFunction.set(nn,mm,r0,d0);
    }
    if (mm!=0 || nn%2!=0 || switchingFunction.get_d0()!=0.0) {
      switchSettings = ::myACC::rationalData<float>(natA,
                       natB,
                       nn,
                       mm,
                       1.0/switchingFunction.get_r0(),
                       switchingFunction.get_d0(),
                       switchingFunction.get_dmax());
    } else {
      switchSettings = ::myACC::switchData<float>(natA,
                       natB,
                       nn,
                       mm,
                       1.0/switchingFunction.get_r0(),
                       switchingFunction.get_dmax());
    }

    std::visit([](auto& data) {
      using T = std::decay_t<decltype(data)>;
      float stretch=1.0;
      float shift=0.0;
      if constexpr (std::is_same_v<T, ::myACC::switchData<float>>) {
        std::tie(stretch,shift) = ::myACC::getShiftAndStretch<::myACC::calculatorReducedRational<float>>(data);
      } else if constexpr (std::is_same_v<T, ::myACC::rationalData<float>>) {
        std::tie(stretch,shift) = ::myACC::getShiftAndStretch<::myACC::calculatorRational<float>>(data);
      }/* else {
        static_assert(false, "non-exhaustive visitor!");
      }*/
      data.stretch=stretch;
      data.shift=shift;
    },
    switchSettings);
  }
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

template <typename mycalculator, typename dataContainer,typename T=typename dataContainer::precision>
std::pair <T,PLMD::wFloat::Vector<T>> switchAlltoAll(unsigned i,
                                   const std::vector<PLMD::wFloat::Vector<T>>& positions,
                                   const std::vector<PLMD::AtomNumber> & reaIndexes,
                                   std::array<T,9>& myVirial,
const dataContainer c)  {
  auto realIndex_i = reaIndexes[i];
  using v3 = PLMD::wFloat::Vector<T>;

  v3 xyz = positions[i];
  v3 mydev(T(0.0),T(0.0),T(0.0));
  T myNcoord=T(0.0);
  myVirial[0]=T(0.0);
  myVirial[1]=T(0.0);
  myVirial[2]=T(0.0);
  myVirial[3]=T(0.0);
  myVirial[4]=T(0.0);
  myVirial[5]=T(0.0);
  myVirial[6]=T(0.0);
  myVirial[7]=T(0.0);
  myVirial[8]=T(0.0);
#pragma acc loop seq
  for (size_t j = 0; j < c.natA; j++) {
    if(realIndex_i==reaIndexes[j]) {
      continue;
    }
    const v3 d=positions[j]-xyz;
    const T dsq=d.modulo2();

    const auto [t,dfunc ]=mycalculator::calculateSqr(dsq,c);
    myNcoord +=t;

    const v3 td = -dfunc * d;
    mydev += td;

    if(i>j) {
      myVirial[0]+=td[0]*d[0];
      myVirial[1]+=td[0]*d[1];
      myVirial[2]+=td[0]*d[2];
      myVirial[3]+=td[1]*d[0];
      myVirial[4]+=td[1]*d[1];
      myVirial[5]+=td[1]*d[2];
      myVirial[6]+=td[2]*d[0];
      myVirial[7]+=td[2]*d[1];
      myVirial[8]+=td[2]*d[2];
    }

  }
  return {myNcoord,mydev};
}

// calculator
void CoordinationACC::calculate() {
  //I should exploit this:
// assert(sizeof(PLMD::AtomNumber)==sizeof(unsigned);
//since AtomNumber wraps an unsigned
  std::vector<unsigned> atomNumbers(getNumberOfAtoms());

  double ncoord=0.;
  Tensor boxDev;
  std::vector<Vector> deriv(getNumberOfAtoms());
  std::vector<PLMD::AtomNumber> reaIndexes=getAbsoluteIndexes();
  std::vector<PLMD::Vector> positions=getPositions();
  std::visit([&](auto data) {
    using T = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<T, ::myACC::switchData<float>>) {
      ncoord=PLMD::parallel::accumulate_sumOP(positions,
                                              reaIndexes,
                                              deriv,
                                              boxDev,
                                              data,
                                              switchAlltoAll<::myACC::calculatorReducedRational<float>,T>);
    } else if constexpr (std::is_same_v<T, ::myACC::rationalData<float>>) {
      ncoord=PLMD::parallel::accumulate_sumOP(positions,
                                              reaIndexes,
                                              deriv,
                                              boxDev,
                                              data,
                                              switchAlltoAll<::myACC::calculatorRational<float>,T>);
    }
  },switchSettings);

  for(unsigned i=0; i<deriv.size(); ++i) {
    setAtomsDerivatives(i,deriv[i]);
  }
  setValue           (ncoord/2.0);
  setBoxDerivatives  (boxDev);

}
} // namespace colvar
}//namespace PLMD
