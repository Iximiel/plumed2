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

#include <torch/torch.h>
#include <torch/script.h>


#include "core/ActionRegister.h"
#include "tools/Communicator.h"
#include "core/PlumedMain.h"
#include "colvar/Colvar.h"

#include <limits>
#include <functional>
#include <type_traits>

#include <iostream>
// #define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'
// #define plotsize(name) std::cerr << __LINE__ << #name ": "<< name.sizes()<< '\n';

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
} //namespace GPU

namespace colvar {
/*
constexpr auto kUInt8 = at::kByte;
constexpr auto kInt8 = at::kChar;
constexpr auto kInt16 = at::kShort;
constexpr auto kInt32 = at::kInt;
constexpr auto kInt64 = at::kLong;
constexpr auto kFloat16 = at::kHalf;
constexpr auto kFloat32 = at::kFloat;
constexpr auto kFloat64 = at::kDouble;
*/

template <typename T>
inline constexpr torch::Dtype getType() {
  //torch requires c++17, so I can make this full c++17
  if constexpr (std::is_same_v<T,float>) {
    return  torch::kFloat32;
  } else if constexpr (std::is_same_v<T,std::int32_t>) {
    return  torch::kInt32;
  } else if constexpr (std::is_same_v<T,std::int64_t>) {
    return  torch::kInt64;
  } else if constexpr (std::is_same_v<T,double>) {
    return  torch::kFloat64;
  }
}

//MEMORY MANAGEMENT//
//deduction guides
template<typename T> struct DataInterface;
//for VectorGeneric
template <unsigned n>
DataInterface(PLMD::VectorGeneric<n>)->DataInterface<double>;
template <unsigned n>
DataInterface(std::vector<PLMD::VectorGeneric<n>>)->DataInterface<double>;
//for TensorGeneric
template <unsigned n, unsigned m>
DataInterface(TensorGeneric<n, m>)->DataInterface<double>;
template <unsigned n, unsigned m>
DataInterface(std::vector<TensorGeneric<n, m>>)->DataInterface<double>;

template<typename T>
struct DataInterface {
  // NOT owning pointer
  //const prt to avoid unwanted changes
  T * const ptr;
  size_t const size;
  DataInterface() = delete;

  template <unsigned n>
  explicit DataInterface (PLMD::VectorGeneric<n> &v, size_t s=1)
    : DataInterface (v[0],s*n) {}

  template <unsigned n, unsigned m>
  explicit DataInterface (PLMD::TensorGeneric<n, m> &v, size_t s=1)
    :DataInterface (v[0][0],s*m*n) {}

  template <unsigned n>
  explicit DataInterface (std::vector<PLMD::VectorGeneric<n>> &vt)
    : DataInterface (vt[0],vt.size()) {}

  template <unsigned n, unsigned m>
  explicit DataInterface (std::vector<PLMD::TensorGeneric<n,m>> &vt)
    : DataInterface (vt[0],vt.size()) {}

  explicit DataInterface (std::vector<T> &vt) : DataInterface (vt[0],vt.size()) {}

  template <size_t N>
  explicit DataInterface (std::array<T,N> &vt) : DataInterface (vt[0],N) {}

  explicit DataInterface (T& v, size_t n=1) : ptr(&v),size(n) {};
};

//loads the data, with conversion if necessary,
//convertFrom is deduced from DataInterface
template <typename convertTo,typename convertFrom>
inline torch::Tensor convertToDevice(DataInterface<convertFrom> dataInterface,
                                     torch::DeviceType device) {
  if constexpr (std::is_same_v<convertTo,convertFrom>) {
    //skipping an unuseful  conversion
    //(I do not kno if the API automagically skips it)
    return torch::from_blob(dataInterface.ptr, {dataInterface.size},
                            getType<convertFrom>()).to(device);
  } else {
    return torch::from_blob(dataInterface.ptr, {dataInterface.size},
                            getType<convertFrom>())
           .to(getType<convertTo>()).to(device);
  }
}

//directly loads the data, without conversion
//works like convertToDevice, but no need to specify explicitly the template
//T is deduced from DataInterface
template <typename T>
inline torch::Tensor loadToDevice(DataInterface<T> dataInterface,
                                  torch::DeviceType device) {
  return convertToDevice<T,T>(dataInterface,device);
}

template <typename calculateFloat>
torch::Tensor setPositions(double* const data, const size_t size,
                           torch::DeviceType device) {
  return torch::from_blob(data, {size,3}, torch::kFloat64)
         .transpose(0,1).to(getType<calculateFloat>()).to(device);
}

template <typename convertTo>
inline void convertFromDevice(const torch::Tensor& origin,
                              DataInterface<convertTo> destination) {
  unsigned const size = origin.numel();
  if(size > destination.size ) {
    plumed_merror("PLMD->AF::getToHost(): destination cannot contain all the data ("+
                  std::to_string(size) + " > " + std::to_string(destination.size) +")");
  }
  torch::Tensor buffer;
  //We need the reshape makes the data in the correct order(!?),
  //I think transpose do not phisically move the data
  //in the contained array and that breaks the memcopy
  if(getType<convertTo>() != buffer.scalar_type()) {
    buffer = origin.detach().to(torch::kCPU).to(getType<convertTo>())
             .reshape({size});
  } else {
    buffer = origin.detach().to(torch::kCPU).reshape({size});

  }
  std::copy(buffer.data_ptr<double>(), buffer.data_ptr<double>() + size, destination.ptr);
  // torch::from_blob(destination.ptr, {destination.size}, getType<double>())  = buffer;
}
//END OF MEMORY MANAGEMENT//

template<typename T>
inline T pbcClamp(const T& x) {
  return x-torch::floor(x+0.5);
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

/*****************************Implementation starts here***********************/
template <typename calculateFloat>
std::pair<torch::Tensor,torch::Tensor> fastRationalN2M(
  const torch::Tensor& distSquared,
  const rationalSwitchParameters<calculateFloat> switchingParameters) {
  auto NN = switchingParameters.nn / 2;
  auto rdistSQ = distSquared * switchingParameters.invr0_2;
  auto rNdist = torch::pow(rdistSQ, NN - 1);
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
std::pair<torch::Tensor,torch::Tensor> fastRational(
  const torch::Tensor& distSquared,
  const rationalSwitchParameters<calculateFloat> switchingParameters) {
  auto NN = switchingParameters.nn / 2;
  auto MM = switchingParameters.mm / 2;
  auto rdistSQ = distSquared * switchingParameters.invr0_2;
  auto rNdist = torch::pow(rdistSQ, NN - 1);
  auto rMdist = torch::pow(rdistSQ, MM - 1);
  auto den = calculateFloat(1.0) / (calculateFloat(1.0) - rdistSQ * rMdist);
  auto res =  (calculateFloat(1.0) - rdistSQ * rNdist)*den;

  //need to use select
  auto dfunc = (MM * rMdist * den * res -NN * rNdist * den )
               * ( 2.0 * switchingParameters.invr0_2 * switchingParameters.stretch);

  return std::make_pair(
           res * switchingParameters.stretch + switchingParameters.shift,
           dfunc);
}

template <typename calculateFloat> class TorchCoordination : public Colvar {
  static auto constexpr myDtype=getType<calculateFloat>();
  // torch::TensorOptions().device(device_t_).dtype(torch::kFloat32);
  torch::TensorOptions tensorOptions =torch::TensorOptions();
  torch::DeviceType myDevice;
  std::tuple<calculateFloat,torch::Tensor,torch::Tensor> work(torch::Tensor& diff,
      torch::Tensor& trueindexes);

  PLMD::GPU::ortoPBCs<calculateFloat> myPBC;
  unsigned atomsInA = 0;
  unsigned atomsInB = 0;
  int  deviceid=-1;

  PLMD::colvar::rationalSwitchParameters<calculateFloat> switchingParameters;

  enum class calculationMode { self, dual, pair, none };
  calculationMode mode = calculationMode::none;
  std::function<
  std::pair<torch::Tensor,torch::Tensor> (const torch::Tensor&,
                                          const rationalSwitchParameters<calculateFloat> )
  > switching;
  bool pbc{true};
public:
  explicit TorchCoordination (const ActionOptions &);
  virtual ~TorchCoordination()=default;
  // active methods:
  static void registerKeywords (Keywords &keys);
  void calculate() override;
};

using TorchCoordination_d = TorchCoordination<double>;
using TorchCoordination_f = TorchCoordination<float>;
PLUMED_REGISTER_ACTION (TorchCoordination_d, "TORCHCOORDINATION")
PLUMED_REGISTER_ACTION (TorchCoordination_f, "TORCHCOORDINATIONFLOAT")

template <typename calculateFloat>
void TorchCoordination<calculateFloat>::registerKeywords (Keywords &keys) {
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
TorchCoordination<calculateFloat>::TorchCoordination (const ActionOptions &ao)
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

// parse("DEVICEID",deviceid);
  if(comm.Get_rank()==0) {
    if (/*gpu_ && */torch::cuda::is_available()) {
      myDevice = torch::kCUDA;
    } else {
      myDevice = torch::kCPU;
      // gpu_ = false;
    }
  }
  tensorOptions =torch::TensorOptions().device(myDevice).dtype(myDtype);
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
      std::array<double,2> inputs = {0.0, dmax};
      //fastRational* gets the square as input
      inputs[1] *= inputs[1];
      torch::Tensor AFinputs=convertToDevice<calculateFloat>(
                               DataInterface(inputs),myDevice);
      auto [res, dfunc] = switching(AFinputs,switchingParameters);
      std::array<double,2> resZeroMax;
      convertFromDevice(res,DataInterface(resZeroMax));

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
}


template <typename calculateFloat>
inline std::tuple<calculateFloat,torch::Tensor,torch::Tensor>
TorchCoordination<calculateFloat>::work(torch::Tensor& diff,
                                        torch::Tensor& keys
                                       ) {
  if(pbc) {
    std::array<calculateFloat,3> vals{myPBC.X.val,myPBC.Y.val,myPBC.Z.val},
        invs{myPBC.X.inv,myPBC.Y.inv,myPBC.Z.inv};
    auto tvals = loadToDevice(DataInterface(vals), myDevice);
    auto tinvs = loadToDevice(DataInterface(invs), myDevice);
    if(diff.dim()==2) { //pair
      tvals=tvals.reshape({3,1});
      tinvs=tinvs.reshape({3,1});
    } else {
      tvals=tvals.reshape({3,1,1});
      tinvs=tinvs.reshape({3,1,1});
    }

    diff = pbcClamp(diff * tinvs) * tvals;
  }
  auto ddistSQ = (diff*diff).sum(0,true);
  keys &= (ddistSQ < switchingParameters.dmaxSQ);
  auto [res,dev] = switching (ddistSQ,switchingParameters);

  auto t = (keys*res).sum()
           .to(torch::kCPU)
           .to(torch::kFloat64);
  //my compiler refuses to use the .data_ptr<double>(), so here's the workaround...
  // double coord = t.item<double>();
  // double coord =*t.data_ptr<double>();
  double coord = *static_cast<double*>(t.data_ptr());
  dev = dev * diff * keys;
  dev *= keys;
  auto AFvirial=torch::zeros({9},tensorOptions);
  {
    using namespace torch::indexing;
    for(int i=0,k=0; i<3; ++i) {
      for(int j=0; j<3; ++j) {
        AFvirial.index_put_({k},-(dev.index({i})* diff.index({j})).sum());
        ++k;
      }
    }
  }

  return std::make_tuple(coord,dev,AFvirial);
}

template <typename calculateFloat>
void TorchCoordination<calculateFloat>::calculate() {
  // std::cerr << getStep() <<"\n";
  if (pbc) {
    makeWhole();
    auto box = getBox();

    myPBC.X = box (0, 0);
    myPBC.Y = box (1, 1);
    myPBC.Z = box (2, 2);
  }
  std::vector <PLMD::Vector> inputs = getPositions();
  auto derivativeA = std::vector<Vector> (atomsInA);
  auto derivativeB = std::vector<Vector> (atomsInB);
  double coordination;
  PLMD::Tensor virial;
  std::vector<int> trueIndexesA (atomsInA);
  for (size_t i = 0; i < atomsInA; ++i) {
    trueIndexesA[i] = getAbsoluteIndex (i).index();
  }
  std::vector<int> trueIndexesB (atomsInB);
  for (size_t i = 0; i < atomsInB; ++i) {
    trueIndexesB[i] = getAbsoluteIndex (atomsInA+i).index();
  }

  switch (mode) {
  case calculationMode::self: {
    torch::Tensor posA=convertToDevice<calculateFloat>(DataInterface(inputs),
                       myDevice)
                       .reshape({atomsInA,3,1})
                       .transpose(0,1)
                       .tile({1,1,atomsInA});
    torch::Tensor posB=convertToDevice<calculateFloat>(
                         DataInterface(inputs),myDevice)
                       .reshape({atomsInA,1,3})
                       .transpose(0,2)
                       .tile({1,atomsInA,1});
    auto diff = posB-posA;
    torch::Tensor indexesA=loadToDevice(DataInterface(trueIndexesA),myDevice)
                           .reshape({1,atomsInA,1})
                           .tile({1,1,atomsInA});
    torch::Tensor indexesB=loadToDevice<int>(DataInterface(trueIndexesA),myDevice)
                           .reshape({1,1,atomsInA})
                           .tile({1,atomsInA,1});
    auto trueindexes = indexesA != indexesB;

    auto[coord, dev, storedVirial] = work(diff,trueindexes);

    coordination = 0.5*coord;
    convertFromDevice( 0.5*storedVirial, DataInterface(virial));
    convertFromDevice( dev.sum(1).transpose(1,0), DataInterface(derivativeA));

  }
  break;
  case calculationMode::dual: {
    auto posA=setPositions<calculateFloat>(&inputs[0][0], atomsInA,myDevice)
              .reshape({3,atomsInA,1})
              .tile({1,1,atomsInB});
    auto posB=setPositions<calculateFloat>(&inputs[atomsInA][0], atomsInB,myDevice)
              .reshape({3,1,atomsInB})
              .tile({1,atomsInA,1});
    auto diff = posB-posA;
    torch::Tensor indexesA=loadToDevice(DataInterface(trueIndexesA),myDevice)
                           .reshape({1,atomsInA,1})
                           .tile({1,1,atomsInB});
    torch::Tensor indexesB=loadToDevice(DataInterface(trueIndexesB),myDevice)
                           .reshape({1,1,atomsInB})
                           .tile({1,atomsInA,1});
    auto trueindexes = indexesA != indexesB;

    auto[coord, dev, storedVirial] = work(diff,trueindexes);

    coordination = coord;
    convertFromDevice( storedVirial, DataInterface(virial));
    convertFromDevice( -dev.sum(2).transpose(1,0), DataInterface(derivativeA));
    convertFromDevice( dev.sum(1).transpose(1,0), DataInterface(derivativeB));

  }
  break;
  case calculationMode::pair: {
    auto posA=setPositions<calculateFloat>(&inputs[0][0], atomsInA,myDevice);
    auto posB=setPositions<calculateFloat>(&inputs[atomsInA][0], atomsInB,myDevice);
    posB -= posA;
    auto trueindexes =
      (loadToDevice(DataInterface(trueIndexesA),myDevice)
       != loadToDevice(DataInterface(trueIndexesB),myDevice))
      .reshape({1,atomsInA});

    auto [res, dev, storedVirial] = work(posB, trueindexes);

    coordination = res;
    convertFromDevice(storedVirial, DataInterface(virial));
    convertFromDevice(-dev.transpose(1,0), DataInterface(derivativeA));
    convertFromDevice( dev.transpose(1,0), DataInterface(derivativeB));
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
