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
//(cd ../../plugins/torchCoord && make ) && cp ../../plugins/torchCoord/TorchCoordination.so . && ./doperf.sh
#include <torch/torch.h>
#include <torch/script.h>


#include "core/ActionRegister.h"
#include "tools/Communicator.h"
#include "core/PlumedMain.h"
#include "colvar/Colvar.h"

#include <limits>
#include <functional>

#include <thread>
#include <chrono>

#include <iostream>
#define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'
#define plotsize(name) std::cerr << __LINE__ << #name ": "<< name.sizes()<< '\n';
#define sleeper(millis) std::this_thread::sleep_for (std::chrono::milliseconds(millis));

#define sleeper(...)
#define vdbg(...)
#define plotsize(...)
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
template<typename T>
struct DataInterface {
  // NOT owning pointer
  T *ptr = nullptr;
  size_t size = 0;
  DataInterface() = delete;

  explicit DataInterface (std::vector<T> &vt) : DataInterface (vt[0]) {
    size *= vt.size();
  }
  template <size_t N>
  explicit DataInterface (std::array<T,N> &vt) : DataInterface (vt[0]) {
    size *= N;
  }
  explicit DataInterface (T& v) : ptr(&v),size(1) {};

  explicit DataInterface (T& v, size_t n) : ptr(&v),size(n) {};
};

template <unsigned n>
inline DataInterface<double> toDataInterface(
  std::vector<PLMD::VectorGeneric<n>> &vg) {
  return DataInterface<double> (vg[0][0], n*vg.size());
}

template <unsigned n, unsigned m>
inline DataInterface<double> toDataInterface(
  std::vector<TensorGeneric<n, m> > &tg) {
  return DataInterface<double> (tg[0][0][0], m*n*tg.size());
}

template <unsigned n, unsigned m>
inline DataInterface<double> toDataInterface(
  TensorGeneric<n, m> &tg) {
  return DataInterface<double> (tg[0][0], m*n);
}

//loads the data, with conversion if necessary,
//convertFrom is deduced from DataInterface
template <typename convertTo,typename convertFrom>
inline torch::Tensor convertToDevice(DataInterface<convertFrom> dataInterface,
                                     torch::DeviceType device) {
  if constexpr (std::is_same_v<convertTo,convertFrom>) {
    //skipping an unuseful  conversion
    //(I do not know if the API automagically skips it)
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
  calculateFloat invr0 = 1.0; // r0=1
  calculateFloat d0 = 0.0;
  calculateFloat stretch = 1.0;
  calculateFloat shift = 0.0;
  int nn = 6;
  int mm = 12;
};


// inline
// torch::autograd::Variable fastpow(torch::autograd::Variable base, unsigned exp) {
//   auto result = torch::ones_like(base);
//   //this does not passes the tests because base gets updated also in the caller
//   while (exp) {
//     if (exp & 1)
//       result *= base;
//     exp >>= 1;
//     base *= base;
//   }
//   vdbg(result.data_ptr());
//   vdbg(base.data_ptr());
//   vdbg(t.data_ptr());
//   return result;
// }


template <typename calculateFloat>
class fastRationalN2MF : public torch::autograd::Function<fastRationalN2MF<calculateFloat>> {
public:


  static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx,
      const rationalSwitchParameters<calculateFloat> switchingParameters,
      torch::autograd::Variable rdistSQ/*distSquared*/
                                               ) {
    auto NN = switchingParameters.nn / 2;
    rdistSQ *= switchingParameters.invr0_2;
    auto rNdist = torch::pow(rdistSQ, NN - 1);
    // auto rNdist = fastpow(rdistSQ, NN - 1);
    auto res =  calculateFloat(1.0) / (calculateFloat(1.0) + rdistSQ * rNdist);

    rNdist *= (-NN * 2.0
               * switchingParameters.invr0_2 * switchingParameters.stretch)
              * res * res;
    res *= switchingParameters.stretch;
    res += switchingParameters.shift;
    return {res,
            rNdist};

    //  // Save data for backward in context
    //  ctx->saved_data["n"] = n;
    //  var.mul_(2);
    //  // Mark var as modified by inplace operation
    //  ctx->mark_dirty({var});
    //  return {var};
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list
      grad_output) {
    return {grad_output[0]};
  }
};

template <typename calculateFloat>
class rowdot : public torch::autograd::Function<rowdot<calculateFloat>> {
public:

  static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx,
      torch::autograd::Variable vec,torch::autograd::Variable key, calculateFloat myMax) {
    auto dot = vec[0]*vec[0]
               +vec[1]*vec[1]
               +vec[2]*vec[2];
    auto mykey=key.bitwise_and(dot<myMax);
    return {dot, mykey};
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list
      grad_output) {
    return {grad_output[0]};
  }
};


/*****************************Implementation starts here***********************/


// inline
// torch::Tensor fastpow(torch::Tensor base, unsigned exp) {
//   auto result = torch::ones_like(base);
//   //this does not passes the tests because base gets updated also in the caller
//   while (exp) {
//     if (exp & 1)
//       result *= base;
//     exp >>= 1;
//     base *= base;
//   }
//   vdbg(result.data_ptr());
//   vdbg(base.data_ptr());
//   vdbg(t.data_ptr());
//   return result;
// }

template <typename calculateFloat>
std::pair<torch::Tensor,torch::Tensor> fastRationalN2M(
  torch::Tensor rdistSQ/*distSquared*/,
  const rationalSwitchParameters<calculateFloat> switchingParameters) {

  vdbg(rdistSQ.data_ptr());

  const auto NN = switchingParameters.nn / 2;
  // auto rdistSQ = distSquared * switchingParameters.invr0_2;
  rdistSQ *= switchingParameters.invr0_2;
  vdbg(rdistSQ.data_ptr());

  // auto rNdist = fastpow(rdistSQ, NN - 1);
  auto rNdist = torch::pow(rdistSQ, NN - 1);
  vdbg(rNdist.data_ptr());
  // auto res =  calculateFloat(1.0) / (calculateFloat(1.0) + rdistSQ * rNdist);
  auto res = torch::reciprocal(torch::add(torch::mul(rdistSQ,rNdist),calculateFloat(1.0)));
  // auto res =  calculateFloat(1.0) / (calculateFloat(1.0) + rdistSQ * rNdist);
  //need to use select
  // auto dfunc = (-NN * 2.0
  //               * switchingParameters.invr0_2 * switchingParameters.stretch)
  //              * rNdist * res * res;


//see if is a good idea to use
//addcmul
  rNdist *= res * res *
            (-NN * 2.0 * switchingParameters.invr0_2 * switchingParameters.stretch);
  vdbg(rdistSQ.data_ptr());
  vdbg(res.data_ptr());
  res *= switchingParameters.stretch;
  vdbg(res.data_ptr());
  res+= switchingParameters.shift;
  vdbg(res.data_ptr());
  return std::make_pair(res,rNdist);

}

template <typename calculateFloat>
std::pair<torch::Tensor,torch::Tensor> fastRational(
  torch::Tensor rdistSQ/*distSquared*/,
  const rationalSwitchParameters<calculateFloat> switchingParameters) {
  auto NN = switchingParameters.nn / 2;
  auto MM = switchingParameters.mm / 2;
  // auto rdistSQ = distSquared * switchingParameters.invr0_2;
  rdistSQ *= switchingParameters.invr0_2;
  auto rNdist = torch::pow(rdistSQ, NN - 1);
  auto rMdist = torch::pow(rdistSQ, MM - 1);
  auto den = calculateFloat(1.0) / (calculateFloat(1.0) - rdistSQ * rMdist);
  auto res =  (calculateFloat(1.0) - rdistSQ * rNdist)*den;


  return std::make_pair(
           res * switchingParameters.stretch + switchingParameters.shift,
           (MM * rMdist * den * res - NN * rNdist * den )
           * ( 2.0 * switchingParameters.invr0_2 * switchingParameters.stretch));
}

template <typename calculateFloat>
std::pair<torch::Tensor,torch::Tensor> cosSwitch(
  torch::Tensor dist/*distSquared*/,
  const rationalSwitchParameters<calculateFloat> switchingParameters) {

  dist=torch::sqrt(dist);

  //D0=0
  // rdist = (distance-d0)/r0
  auto res = dist*(switchingParameters.invr0*PLMD::pi);
  auto dfunc = (-0.5 * PLMD::pi  * switchingParameters.invr0_2//*switchingParameters.stretch
               )* torch::sin ( res );
  
  auto tores = 0.5*(torch::cos(res)+1);

  dfunc/=dist;
//there are some division by 0;
  dfunc=dfunc.nan_to_num(0.0);
//with current implementation the swich can ve ingnored
  // res *= switchingParameters.stretch;
  // res+= switchingParameters.shift;
  // dfunc*=;
  plotsize(tores);
  plotsize(dfunc);
  plotsize(dist);
  {
    using namespace torch::indexing;
    if(dfunc.dim()>1) {
      vdbg(dfunc.index({0,1}));
      vdbg(dist.index({0,1}));
      vdbg(res.index({0,1}));
    }
  }
  return std::make_pair(tores,dfunc);
}

template <typename calculateFloat> class TorchCoordination : public Colvar {
  static auto constexpr myDtype=getType<calculateFloat>();
  // torch::TensorOptions().device(device_t_).dtype(torch::kFloat32);
  torch::TensorOptions tensorOptions =torch::TensorOptions();
  torch::DeviceType myDevice;
  std::tuple<calculateFloat,torch::Tensor,PLMD::Tensor> work(torch::Tensor& diff,
      torch::Tensor& trueindexes);

  PLMD::GPU::ortoPBCs<calculateFloat> myPBC;
  unsigned atomsInA = 0;
  unsigned atomsInB = 0;
  int  deviceid=-1;

  PLMD::colvar::rationalSwitchParameters<calculateFloat> switchingParameters;

  enum class calculationMode { self, dual, pair, none };
  calculationMode mode = calculationMode::none;
  std::function<
  std::pair<torch::Tensor,torch::Tensor> (torch::Tensor&,
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
  keys.add("compulsory","SWITCH","rational","rational or cosinus");

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
      log.printf ("  Using a CUDA device with libtorch\n");
      myDevice = torch::kCUDA;
    } else {
      log.printf ("  Using the CPU as libtorch device\n");
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
    {
      std::string switchType;
      parse("SWITCH",switchType);
      if(switchType=="cosinus") {
        switching = cosSwitch<calculateFloat>;
        dmax=r0_;
      }
    }

    switchingParameters.dmaxSQ = dmax * dmax;
    switchingParameters.invr0 = 1.0 / r0_;
    switchingParameters.invr0_2 = switchingParameters.invr0 * switchingParameters.invr0;
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


  vdbg(switchingParameters.dmaxSQ);
  vdbg(switchingParameters.invr0_2);
  vdbg(switchingParameters.invr0);
  vdbg(switchingParameters.d0);
  vdbg(switchingParameters.stretch);
  vdbg(switchingParameters.shift);
  vdbg(switchingParameters.nn);
  vdbg(switchingParameters.mm);

  log << "  contacts are counted with cutoff (dmax)="
      << sqrt (switchingParameters.dmaxSQ)
      << ",\nwith a rational switch with parameters: d0=0.0, r0="
      << 1.0 / sqrt (switchingParameters.invr0_2)
      << ", N=" << switchingParameters.nn << ", M=" << switchingParameters.mm
      << ".\n";
}

template <typename calculateFloat>
inline std::tuple<calculateFloat,torch::Tensor,PLMD::Tensor>
TorchCoordination<calculateFloat>::work(torch::Tensor& diff,
                                        torch::Tensor& keys
                                       ) {
  sleeper(2);
  if(pbc) {
    std::array<calculateFloat,3> vals{myPBC.X.val,myPBC.Y.val,myPBC.Z.val},
        invs{myPBC.X.inv,myPBC.Y.inv,myPBC.Z.inv};
    auto tvals = loadToDevice(DataInterface(vals), myDevice).reshape({3,1});
    auto tinvs = loadToDevice(DataInterface(invs), myDevice).reshape({3,1});
    diff = torch::mul(pbcClamp(torch::mul(diff, tinvs)),tvals);
  }
  sleeper(1);
  // auto nOperations = diff.numel()/3;
  // auto ddistSQ = torch::mul(diff,diff).sum(0,true);
  // auto getData = rowdot<calculateFloat>::apply(diff, keys, switchingParameters.dmaxSQ);
  // auto ddistSQ = getData[0].reshape({1,nOperations});
  // vdbg(ddistSQ.data_ptr());
  auto ddistSQ = (diff*diff).sum(0,true);
  keys &= (ddistSQ < switchingParameters.dmaxSQ);

  // auto ddistSQ = torch::mul(diff,diff).sum(0,true);

  // keys= getData[1];
  sleeper(3);
  // torch::Tensor res;
  // std::tie(tmp,ddistSQ) =
  plotsize(ddistSQ);
  vdbg("switching");
  auto [res,dev] =
    switching (ddistSQ,switchingParameters);
  plotsize(res);
  plotsize(dev);
  plotsize(ddistSQ);
  vdbg("returned");
  vdbg(ddistSQ.data_ptr());
  vdbg(res.data_ptr());
  sleeper(2);


  dev *= keys;
  dev = dev * diff;
  // ddistSQ =keys * ddistSQ * diff;


  sleeper(1);
  PLMD::Tensor virial;
  // auto t= -torch::kron(ddistSQ,diff);
  // plotsize(t);
  // convertFromDevice(t.sum(1),DataInterface(virial[0][0],9))
  // for(int i=0; i<3; ++i) {
  //   for(int j=0; j<3; ++j) {
  //     convertFromDevice(-torch::dot(dev.index({i}), diff.index({j})),
  //     // convertFromDevice(-torch::dot(ddistSQ.index({i}), diff.index({j})),
  //                       DataInterface(virial[i][j]));
  //   }
  // }
  auto gpuvirial=torch::zeros({9},tensorOptions);
  {
    using namespace torch::indexing;
    for(int i=0,k=0; i<3; ++i) {
      for(int j=0; j<3; ++j) {
        // gpuvirial.index_put_({k},-(dev.index({i})* diff.index({j})).sum());
        gpuvirial.index_put_({k},-(dev.index({i}).dot(diff.index({j}))));
        ++k;
      }
    }
  }
  convertFromDevice(gpuvirial,DataInterface(virial[0][0],9))

  // {
  //   convertFromDevice(-torch::dot(ddistSQ.index({0}), diff.index({0})),DataInterface(virial[0][0]));
  //   convertFromDevice(-torch::dot(ddistSQ.index({0}), diff.index({1})),DataInterface(virial[0][1]));
  //   convertFromDevice(-torch::dot(ddistSQ.index({0}), diff.index({2})),DataInterface(virial[0][2]));
  //   convertFromDevice(-torch::dot(ddistSQ.index({1}), diff.index({0})),DataInterface(virial[1][0]));
  //   convertFromDevice(-torch::dot(ddistSQ.index({1}), diff.index({1})),DataInterface(virial[1][1]));
  //   convertFromDevice(-torch::dot(ddistSQ.index({1}), diff.index({2})),DataInterface(virial[1][2]));
  //   convertFromDevice(-torch::dot(ddistSQ.index({2}), diff.index({0})),DataInterface(virial[2][0]));
  //   convertFromDevice(-torch::dot(ddistSQ.index({2}), diff.index({1})),DataInterface(virial[2][1]));
  //   convertFromDevice(-torch::dot(ddistSQ.index({2}), diff.index({2})),DataInterface(virial[2][2]));
  // }

  sleeper(1);
  double coord;

  convertFromDevice(torch::sum(res*keys), DataInterface(coord));
  sleeper(3);
  return std::make_tuple(coord,dev,virial);
  // return std::make_tuple(coord,ddistSQ,virial);
}

template <typename calculateFloat>
void TorchCoordination<calculateFloat>::calculate() {
  //deactivating autograd should(?) accelerates
  auto deactivateAutoGrad=c10::InferenceMode(true);
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
    torch::Tensor posA=convertToDevice<calculateFloat>(toDataInterface(inputs),
                       myDevice)
                       .reshape({atomsInA,3,1})
                       .transpose(0,1)
                       .tile({1,1,atomsInA});
    torch::Tensor posB=convertToDevice<calculateFloat>(
                         toDataInterface(inputs),myDevice)
                       .reshape({atomsInA,1,3})
                       .transpose(0,2)
                       .tile({1,atomsInA,1});
    posB-=posA;
    posB=posB.reshape({3,atomsInA*atomsInA});

    torch::Tensor indexesA=loadToDevice(DataInterface(trueIndexesA),myDevice)
                           .reshape({atomsInA,1})
                           .tile({1,atomsInA});
    torch::Tensor indexesB=loadToDevice<int>(DataInterface(trueIndexesA),myDevice)
                           .reshape({1,atomsInA})
                           .tile({atomsInA,1});
    auto trueindexes = (indexesA != indexesB).reshape({1,atomsInA*atomsInA});

    auto[coord, dev, storedVirial] = work(posB,trueindexes);
    dev = dev.reshape({3,atomsInA,atomsInA});
    coordination = 0.5*coord;
    // convertFromDevice( 0.5*storedVirial, toDataInterface(virial));
    virial = 0.5*storedVirial;
    convertFromDevice( dev.sum(1).transpose(1,0), toDataInterface(derivativeA));

  }
  break;
  case calculationMode::dual: {
    auto posA=setPositions<calculateFloat>(&inputs[0][0], atomsInA,myDevice)
              .reshape({3,atomsInA,1})
              .tile({1,1,atomsInB});
    auto posB=setPositions<calculateFloat>(&inputs[atomsInA][0], atomsInB,myDevice)
              .reshape({3,1,atomsInB})
              .tile({1,atomsInA,1});
    posB-=posA;
    posB=posB.reshape({3,atomsInB*atomsInA});
    torch::Tensor indexesA=loadToDevice(DataInterface(trueIndexesA),myDevice)
                           .reshape({atomsInA,1})
                           .tile({1,atomsInB});
    torch::Tensor indexesB=loadToDevice(DataInterface(trueIndexesB),myDevice)
                           .reshape({1,atomsInB})
                           .tile({atomsInA,1});
    auto trueindexes = (indexesA != indexesB).reshape({1,atomsInA*atomsInB});
    auto[coord, dev, storedVirial] = work(posB,trueindexes);
    dev = dev.reshape({3,atomsInA,atomsInB});
    coordination = coord;
    // convertFromDevice( storedVirial, toDataInterface(virial));
    virial = storedVirial;
    convertFromDevice( -dev.sum(2).transpose(1,0), toDataInterface(derivativeA));
    convertFromDevice( dev.sum(1).transpose(1,0), toDataInterface(derivativeB));

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
    // convertFromDevice(storedVirial, toDataInterface(virial));
    virial = storedVirial;
    convertFromDevice(-dev.transpose(1,0), toDataInterface(derivativeA));
    convertFromDevice( dev.transpose(1,0), toDataInterface(derivativeB));
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
