/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2024 The plumed team
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

#ifndef __PLUMED_HAS_ARRAYFIRE
#error "This CV can only be built if the current plumed2 has found arrayfire"
#endif //__PLUMED_HAS_ARRAYFIRE

#include <arrayfire.h>
#ifdef __PLUMED_HAS_ARRAYFIRE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <af/cuda.h>
namespace afdevice=afcu;
#elif __PLUMED_HAS_ARRAYFIRE_OCL
#include <af/opencl.h>
namespace afdevice=afocl;
#else
//this is a small workaround to use less #ifdef ;)
namespace afdevice {
int getNativeId(int const i) { return i;}
}
#endif


#include "core/ActionRegister.h"
#include "tools/Communicator.h"
#include "core/PlumedMain.h"
#include "colvar/Colvar.h"

#include <limits>
#include <functional>

// #include <iostream>
// #define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'
// #define vdbg(...)
namespace PLMD {
//stolen from cuda implementation
namespace GPU {
template <typename T> struct invData {
  T val = 1.0;
  T inv = 1.0;
  // this makes the `X = x;` work like "X.val=x;X.inv=1/x;"
  // and the compiler will do some inline magic for you
  invData (T const v) : val{v}, inv{T (1.0) / v} {}
  invData &operator= (T const v) {
    val = v;
    inv = T (1.0) / v;
    return *this;
  }
};
template <typename calculateFloat> struct ortoPBCs {
  invData<calculateFloat> X{1.0};
  invData<calculateFloat> Y{1.0};
  invData<calculateFloat> Z{1.0};
};
}
namespace colvar {

//stolen from my CUDA implementation:
struct DataInterface {
  // NOT owning pointer
  double *ptr = nullptr;
  size_t size = 0;
  DataInterface() = delete;

  // VectorGeneric is a "memory map" on an n linear array
  // &vg[0] gets the pointer to the first double in memory
  // C++ vectors align memory so we can safely use the vector variant of
  // DataInterface
  template <unsigned n>
  explicit DataInterface (PLMD::VectorGeneric<n> &vg)
    : ptr (&vg[0]), size (n) {}
  // TensorGeneric is a "memory map" on an n*m linear array
  // &tg[0][0] gets  the pointer to the first double in memory
  // C++ vectors align memory so we can safely use the vector variant of
  // DataInterface
  template <unsigned n, unsigned m>
  explicit DataInterface (PLMD::TensorGeneric<n, m> &tg)
    : ptr (&tg[0][0]), size (n * m) {}
  template <typename T>
  explicit DataInterface (std::vector<T> &vt) : DataInterface (vt[0]) {
    size *= vt.size();
  }
};


template <typename calculateFloat>
constexpr af_dtype getType() {
  return  af_dtype::f32;
}

template <>
constexpr af_dtype getType<double>() {
  return  af_dtype::f64;
}

template <typename calculateFloat>
inline af::array setPositions(const double* const data, const size_t size) {
  std::vector<float> posi(3*size);
  for (unsigned i=0; i<size; ++i) {
    posi[3*i]   = static_cast<float>(data[3*i] );
    posi[3*i+1] = static_cast<float>(data[3*i+1]);
    posi[3*i+2] = static_cast<float>(data[3*i+2]);
  }
  return af::array(3, size, &posi.front()).T();
}

template<>
inline af::array setPositions<double>(const double* const data, const size_t size) {
  return af::array(3, size, data).T();
}


template <typename calculateFloat>
inline void getToHost(const af::array& input,
                      DataInterface destination) {
  unsigned const size = input.elements();
  if(size > destination.size ) {
    plumed_merror("PLMD->AF::getToHost(): destination cannot contain all the data ("+
                  std::to_string(size) + " > " + std::to_string(destination.size) +")");
  }
  std::vector<float> buffer(size);
  input.host(buffer.data());
  for (unsigned i=0; i<size; ++i) {
    destination.ptr[i] =  static_cast<double>(buffer[i]);
  }
}

template <>
inline void getToHost<double>(const af::array& input,
                              DataInterface destination) {
  unsigned const size = input.elements();
  if(size >(destination.size) ) {
    plumed_merror("PLMD->AF::getToHost(): destination cannot contain all the data ("+
                  std::to_string(size) + " > " + std::to_string(destination.size) +")");
  }
  input.host(destination.ptr);
}


template<typename T>
inline T pbcClamp(const T& x) {
  return x-af::floor(x+0.5);
}

//TODO: this is identical to the plain cuda implementation
template <typename calculateFloat> struct rationalSwitchParameters {
  calculateFloat dmaxSQ = std::numeric_limits<calculateFloat>::max();
  calculateFloat invr0_2 = 1.0; // r0=1
  calculateFloat stretch = 1.0;
  calculateFloat shift = 0.0;
  int nn = 6;
  int mm = 12;
};

template <typename calculateFloat>
std::pair<af::array,af::array> fastRationalN2M(const af::array& distSquared,
    const rationalSwitchParameters<calculateFloat> switchingParameters) {
  auto NN = switchingParameters.nn / 2;
  auto rdistSQ = distSquared * switchingParameters.invr0_2;
  auto rNdist = af::pow(rdistSQ, NN - 1);
  auto res =  calculateFloat(1.0) / (calculateFloat(1.0) + rdistSQ * rNdist);
  //need to use select
  auto dfunc = (-NN * 2.0
                * switchingParameters.invr0_2 * switchingParameters.stretch)
               * rNdist * res * res;

  return std::make_pair(
           res * switchingParameters.stretch + switchingParameters.shift,
           dfunc);
}

template <typename calculateFloat>
std::pair<af::array,af::array> fastRational(const af::array& distSquared,
    const rationalSwitchParameters<calculateFloat> switchingParameters) {
  auto NN = switchingParameters.nn / 2;
  auto MM = switchingParameters.mm / 2;
  auto rdistSQ = distSquared * switchingParameters.invr0_2;
  auto rNdist = af::pow(rdistSQ, NN - 1);
  auto rMdist = af::pow(rdistSQ, MM - 1);
  auto den = calculateFloat(1.0) / (calculateFloat(1.0) - rdistSQ * rMdist);
  auto res =  (calculateFloat(1.0) - rdistSQ * rNdist)*den;

  //need to use select
  auto dfunc = (MM * rMdist * den * res -NN * rNdist * den )
               * ( 2.0 * switchingParameters.invr0_2 * switchingParameters.stretch);

  return std::make_pair(
           res * switchingParameters.stretch + switchingParameters.shift,
           dfunc);
}

template <typename calculateFloat> class FireCoordination : public Colvar {
  std::tuple<calculateFloat,af::array,af::array> work(af::array& diff,
      af::array& trueindexes);

  PLMD::GPU::ortoPBCs<calculateFloat> myPBC;
  unsigned atomsInA = 0;
  unsigned atomsInB = 0;
  int  deviceid=-1;

  PLMD::colvar::rationalSwitchParameters<calculateFloat> switchingParameters;

  enum class calculationMode { self, dual, pair, none };
  calculationMode mode = calculationMode::none;
  std::function<
  std::pair<af::array,af::array> (const af::array&,
                                  const rationalSwitchParameters<calculateFloat> )
  > switching;
  bool pbc{true};
public:
  explicit FireCoordination (const ActionOptions &);
  virtual ~FireCoordination()=default;
  // active methods:
  static void registerKeywords (Keywords &keys);
  void calculate() override;
};

using FireCoordination_d = FireCoordination<double>;
using FireCoordination_f = FireCoordination<float>;
PLUMED_REGISTER_ACTION (FireCoordination_d, "FIRECOORDINATION")
PLUMED_REGISTER_ACTION (FireCoordination_f, "FIRECOORDINATIONFLOAT")

template <typename calculateFloat>
void FireCoordination<calculateFloat>::registerKeywords (Keywords &keys) {
  Colvar::registerKeywords (keys);

  // keys.add ("optional", "THREADS", "The upper limit of the number of threads");
  keys.add ("atoms", "GROUPA", "First list of atoms");
  keys.add ("atoms", "GROUPB", "Second list of atoms, optional");
  keys.addFlag ("PAIR",
                false,
                "Pair only 1st element of the 1st group with 1st element in "
                "the second, etc");

  keys.add (
    "compulsory", "NN", "6", "The n parameter of the switching function ");
  keys.add ("compulsory",
            "MM",
            "0",
            "The m parameter of the switching function; 0 implies 2*NN");
  keys.add ("compulsory", "R_0", "The r_0 parameter of the switching function");
  keys.add (
    "compulsory", "D_MAX", "0.0", "The cut off of the switching function");
  keys.add("compulsory","DEVICEID","-1","Identifier of the GPU to be used");

}
template <typename calculateFloat>
FireCoordination<calculateFloat>::FireCoordination (const ActionOptions &ao)
  : PLUMED_COLVAR_INIT (ao) {
  std::vector<AtomNumber> GroupA;
  parseAtomList ("GROUPA", GroupA);
  std::vector<AtomNumber> GroupB;
  parseAtomList ("GROUPB", GroupB);

  if (GroupB.size() == 0) {
    mode = calculationMode::self;
    atomsInA = GroupA.size();
  } else {
    mode = calculationMode::dual;
    atomsInA = GroupA.size();
    atomsInB = GroupB.size();
    bool dopair = false;
    parseFlag ("PAIR", dopair);
    if (dopair) {
      if (GroupB.size() == GroupA.size()) {
        mode = calculationMode::pair;
      } else {
        error ("GROUPA and GROUPB must have the same number of atoms if the "
               "PAIR keyword is present");
      }
    }
    GroupA.insert (GroupA.end(), GroupB.begin(), GroupB.end());
  }
  if (mode == calculationMode::none) {
    error (
      R"(implementation error in constructor: calculation mode cannot be "none")");
  }
  bool nopbc = !pbc;
  parseFlag ("NOPBC", nopbc);
  pbc = !nopbc;

  // parse ("THREADS", maxNumThreads);
  // if (maxNumThreads <= 0) {
  //   error ("THREADS should be positive");
  // }
  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms (GroupA);

  log.printf ("  \n");
  if (pbc) {
    log.printf ("  using periodic boundary conditions\n");
  } else {
    log.printf ("  without periodic boundary conditions\n");
  }
  if (mode == calculationMode::pair) {
    log.printf ("  with PAIR option\n");
  }
  std::string sw, errors;

  { // loading data to the GPU
    int nn_ = 6;
    int mm_ = 0;

    calculateFloat r0_ = 0.0;
    parse ("R_0", r0_);
    if (r0_ <= 0.0) {
      error ("R_0 should be explicitly specified and positive");
    }

    parse ("NN", nn_);
    parse ("MM", mm_);

    if (mm_ == 0) {
      mm_ = 2 * nn_;
    }
    if (mm_ % 2 != 0 || mm_ % 2 != 0) {
      error (" this implementation only works with both MM and NN even");
    }
    // constexpr auto d0_ = 0.0;
    if(mm_ == 2*nn_) {
      switching=fastRationalN2M<calculateFloat>;
    } else {
      switching=fastRational<calculateFloat>;
    }
    switchingParameters.nn = nn_;
    switchingParameters.mm = mm_;
    switchingParameters.stretch = 1.0;
    switchingParameters.shift = 0.0;

    calculateFloat dmax = 0.0;
    parse ("D_MAX", dmax);
    if (dmax == 0.0) { // TODO:check for a "non present flag"
      // set dmax to where the switch is ~0.00001
      dmax = r0_ * std::pow (0.00001, 1.0 / (nn_ - mm_));
      // ^This line is equivalent to:
      // SwitchingFunction tsw;
      // tsw.set(nn_,mm_,r0_,0.0);
      // dmax=tsw.get_dmax();
      // in plain plumed
    }

    switchingParameters.dmaxSQ = dmax * dmax;
    calculateFloat invr0 = 1.0 / r0_;
    switchingParameters.invr0_2 = invr0 * invr0;
    constexpr bool dostretch = true;
    if (dostretch) {
      std::array<calculateFloat,2> inputs = {0.0, dmax};
      //fastRational* gets the square as input
      inputs[1] *= inputs[1];
      af::array AFinputs(2,1,inputs.data());
      auto [res, dfunc] = switching(AFinputs,switchingParameters);
      std::array<calculateFloat,2> resZeroMax;
      res.host(resZeroMax.data());

      switchingParameters.stretch = 1.0 / (resZeroMax[0] - resZeroMax[1]);
      switchingParameters.shift = -resZeroMax[1] * switchingParameters.stretch;
    }
  }

  checkRead();

  log << "  contacts are counted with cutoff (dmax)="
      << sqrt (switchingParameters.dmaxSQ)
      << ",\nwith a rational switch with parameters: d0=0.0, r0="
      << 1.0 / sqrt (switchingParameters.invr0_2)
      << ", N=" << switchingParameters.nn << ", M=" << switchingParameters.mm
      << ".\n";

  parse("DEVICEID",deviceid);
  if(comm.Get_rank()==0) {
    // if not set try to check the one set by the API
    if(deviceid==-1) deviceid=plumed.getGpuDeviceId();
    // if still not set use 0
    if(deviceid==-1) deviceid=0;
    //afdevice will change to the namespace found by plumed, (see the include clauses)
    af::setDevice(afdevice::getNativeId(deviceid));
    af::info();
  }
}

template <typename calculateFloat>
inline std::tuple<calculateFloat,af::array,af::array>
FireCoordination<calculateFloat>::work(af::array& diff,
                                       af::array& trueindexes
                                      ) {

  if(pbc) {
    diff(af::span,af::span,0) = pbcClamp(diff(af::span,af::span,0) * myPBC.X.inv) * myPBC.X.val;
    diff(af::span,af::span,1) = pbcClamp(diff(af::span,af::span,1) * myPBC.Y.inv) * myPBC.Y.val;
    diff(af::span,af::span,2) = pbcClamp(diff(af::span,af::span,2) * myPBC.Z.inv) * myPBC.Z.val;
  }
  auto ddistSQ = af::sum(diff * diff,2);
  //now we discard the distances that are greater than the limit
  //and the atoms that have the same index
  auto keys = (ddistSQ < switchingParameters.dmaxSQ) - trueindexes;

  auto [res, dfunc] = switching(ddistSQ, switchingParameters);
  calculateFloat coord;
  af::sum(af::sum(keys*res)).host(&coord);

  auto AFderiv = af::select(keys, dfunc * diff, 0.0);

  const unsigned natA = diff.dims()[1];
  const unsigned natB = diff.dims()[0];

  auto AFvirial=af::array(natB,natA,9,getType<calculateFloat>());

  AFvirial(af::span,af::span,0) = AFderiv(af::span,af::span,0) * diff(af::span,af::span,0);
  AFvirial(af::span,af::span,1) = AFderiv(af::span,af::span,0) * diff(af::span,af::span,1);
  AFvirial(af::span,af::span,2) = AFderiv(af::span,af::span,0) * diff(af::span,af::span,2);
  AFvirial(af::span,af::span,3) = AFderiv(af::span,af::span,1) * diff(af::span,af::span,0);
  AFvirial(af::span,af::span,4) = AFderiv(af::span,af::span,1) * diff(af::span,af::span,1);
  AFvirial(af::span,af::span,5) = AFderiv(af::span,af::span,1) * diff(af::span,af::span,2);
  AFvirial(af::span,af::span,6) = AFderiv(af::span,af::span,2) * diff(af::span,af::span,0);
  AFvirial(af::span,af::span,7) = AFderiv(af::span,af::span,2) * diff(af::span,af::span,1);
  AFvirial(af::span,af::span,8) = AFderiv(af::span,af::span,2) * diff(af::span,af::span,2);
  //we constructed an
  //dx*dfunc dx*dfunc dx*dfunc dz*dfunc dz*dfunc dz*dfunc dy*dfunc dy*dfunc dy*dfunc
  //tensor, and we multiply it by the tensor:
  //dx dy dz dx dy dz dx dy dz
  //"external product"
  //no need to lookup/select, already done in  the deriv
  AFvirial = -af::sum(af::sum(AFvirial, 0),1);

  return std::make_tuple(coord,AFderiv,AFvirial);
}

template <typename calculateFloat>
void FireCoordination<calculateFloat>::calculate () {

  double coordination;
  auto derivativeA = std::vector<Vector> (atomsInA);
  auto derivativeB = std::vector<Vector> (atomsInB);


  std::vector<unsigned> trueIndexesA (atomsInA);
  for (size_t i = 0; i < atomsInA; ++i) {
    trueIndexesA[i] = getAbsoluteIndex (i).index();
  }
  std::vector<unsigned> trueIndexesB (atomsInB);
  for (size_t i = 0; i < atomsInB; ++i) {
    trueIndexesB[i] = getAbsoluteIndex (atomsInA+i).index();
  }
  if (pbc) {
    makeWhole();
    auto box = getBox();

    myPBC.X = box (0, 0);
    myPBC.Y = box (1, 1);
    myPBC.Z = box (2, 2);
  }
  PLMD::Tensor virial;
  switch (mode) {
  case calculationMode::self:
  {
    auto posA = setPositions<calculateFloat>(&getPositions()[0][0], atomsInA);
    auto posB = setPositions<calculateFloat>(&getPositions()[0][0], atomsInA);
    posB = af::tile(af::moddims(posB,atomsInA,1,3),1,atomsInA);
    posA = af::tile(af::moddims(posA,1,atomsInA,3),atomsInA,1);
    auto indexesA = af::array(1,atomsInA,trueIndexesA.data());
    auto indexesB = af::array(atomsInA,trueIndexesA.data());
    auto diff = posB - posA;
    auto trueindexes = af::tile(indexesA,atomsInA,1) == af::tile(indexesB,1,atomsInA);

    auto [
      res,
      AFderiv,
      AFvirial
    ] = work(diff, trueindexes);

    getToHost<calculateFloat>(0.5*AFvirial, DataInterface(virial));

    coordination =0.5* res;
    // summing on the second index to not change the sign
    // getToHost<calculateFloat>(-af::sum(AFderiv,0).T(), DataInterface(derivativeA));
    getToHost<calculateFloat>( af::moddims(af::sum(AFderiv,1),atomsInA,3).T(), DataInterface(derivativeA));
  }
  break;
  case calculationMode::dual:
  {
    auto posA = setPositions<calculateFloat>(&getPositions()[0][0], atomsInA);
    auto posB = setPositions<calculateFloat>(&getPositions()[atomsInA][0], atomsInB);
    posB = af::tile(af::moddims(posB,atomsInB,1,3),1,atomsInA);
    posA = af::tile(af::moddims(posA,1,atomsInA,3),atomsInB,1);
    auto diff = posB - posA;

    auto indexesA = af::array(1,atomsInA,trueIndexesA.data());
    auto indexesB = af::array(atomsInB,trueIndexesB.data());
    auto trueindexes = af::tile(indexesA,atomsInB,1) == af::tile(indexesB,1,atomsInA);
    auto [
      res,
      AFderiv,
      AFvirial
    ] = work(diff, trueindexes);

    getToHost<calculateFloat>(AFvirial, DataInterface(virial));

    coordination = res;

    getToHost<calculateFloat>(-af::moddims(af::sum(AFderiv,0),atomsInA,3).T(), DataInterface(derivativeA));
    getToHost<calculateFloat>( af::moddims(af::sum(AFderiv,1),atomsInB,3).T(), DataInterface(derivativeB));
  }
  break;
  case calculationMode::pair:
  { //PAIR
    auto posA = setPositions<calculateFloat>(&getPositions()[0][0], atomsInA);
    auto posB = setPositions<calculateFloat>(&getPositions()[atomsInA][0], atomsInB);
    posB -= posA;
    posB=af::moddims(posB,atomsInA,1,3);
    auto trueindexes =
      af::array(atomsInA,trueIndexesA.data())
      == af::array(atomsInB,trueIndexesB.data());

    auto [
      res,
      AFderiv,
      AFvirial
    ] = work(posB, trueindexes);

    getToHost<calculateFloat>(AFvirial, DataInterface(virial));

    coordination = res;

    getToHost<calculateFloat>(-af::moddims(AFderiv,atomsInA,3).T(), DataInterface(derivativeA));
    getToHost<calculateFloat>( af::moddims(AFderiv,atomsInB,3).T(), DataInterface(derivativeB));
  }
  break;
  case calculationMode::none:
    // throw"this should not have been happened"
    break;
  }

  for(unsigned i=0u; i < atomsInA ; ++i) {
    setAtomsDerivatives (i, derivativeA[i]);
  }

  for(unsigned i=0u; i < atomsInB; ++i) {
    setAtomsDerivatives (atomsInA+i, derivativeB[i]);
  }

  setValue (coordination);
  setBoxDerivatives (virial);
}
} // namespace colvar
} // namespace PLMD
