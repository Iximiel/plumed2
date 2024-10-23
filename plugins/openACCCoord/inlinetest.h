#ifndef myACCinline_hxx
#define myACCinline_hxx
#include <cmath>
#include <iostream>
#include <utility>

#include "plumed/tools/Tools.h"
#include "plumed/tools/AtomNumber.h"

#include "Vector.h"
#include "Tensor.h"
#include "Tools_pow.h"

namespace myACC {

using v3= PLMD::wFloat::Vector<float>;
using tens= PLMD::wFloat::Tensor<float>;


template <typename T>
static inline std::pair<T,T> doReducedRational(const T rdist, const unsigned pow, T result=0.0) {
  const T rNdist=myACC::Tools::fastpow(rdist, pow-1);
  result=1.0/(1.0+rNdist*rdist);
  const T dfunc = -rNdist*result*result*pow;
  return {result,dfunc};
}
template <typename T>
struct calculatorReducedRationalFlexible {
  static std::pair<T,T> calculateSqr(const T distance2,
                                     const T invr0_2,
                                     const T dmax_2,
                                     const T stretch,
                                     const T shift,
                                     const unsigned pow) {
    // int mul = (distance2 <= dmax_2);
    if (distance2 > dmax_2) {
      return {0.0f,0.0f};
    }
    const T rdist = distance2*invr0_2;
    auto [result,dfunc] = doReducedRational(rdist,pow/2);
    dfunc*=2*invr0_2;
    // stretch:
    result=result*stretch+shift;
    dfunc*=stretch;
    return {result,dfunc};
  }
};

template <typename calculator>
std::pair<float,float> getShiftAndStretch(float const invr0_2, float const dmaxsq, const unsigned pow) {
  const float s0=calculator::calculateSqr(0.0f,invr0_2,dmaxsq+1.0,1.0,0.0,pow).first;
  const float sd=calculator::calculateSqr(dmaxsq,invr0_2,dmaxsq+1.0,1.0,0.0,pow).first;

  float const stretch=1.0f/(s0-sd);
  float const shift=-sd*stretch;
  return {shift,stretch};
}

using mycalculator = calculatorReducedRationalFlexible<float>;

template<typename T>
class fastCoordINLINE {
  unsigned natA{0};
  unsigned natB{0};
  unsigned NN{0};
  T invr0_2{1.0};
  T dmaxsq{1.0};
  T shift{0.0};
  T stretch{1.0};
public:
  fastCoordINLINE() = default;
  // fastCoordINLINE(const fastCoordINLINE&) = default;
  // fastCoordINLINE(fastCoordINLINE&&) = default;
  // fastCoordINLINE& operator=(const fastCoordINLINE&) = default;
  // fastCoordINLINE& operator=(fastCoordINLINE&&) = default;
  fastCoordINLINE(unsigned const natA_,
                  unsigned const natB_,
                  unsigned const N,
                  unsigned const M,
                  T const invr0_,
                  T const dmax_)
    : natA(natA_),
      natB(natB_),
      NN(N),
      invr0_2(invr0_*invr0_),
      dmaxsq(dmax_*dmax_) {
    auto [setShift, setStretch] = getShiftAndStretch<mycalculator>(invr0_2, dmaxsq,NN);
    shift = setShift;
    stretch = setStretch;
  }
  T operator()(v3 const xyz,
               unsigned const realIndex_i,
               v3* const positions,
               const unsigned* const reaIndexes) const {
    T myNcoord=0.0;
#pragma acc loop seq
    for (size_t j = 0; j < natA; j++) {
      if(realIndex_i==reaIndexes[j]) {
        continue;
      }
      const v3 d=positions[j]-xyz;
      const float dsq=d.modulo2();
      const auto [t,dfunc ]=mycalculator::calculateSqr(dsq,invr0_2,dmaxsq, stretch,shift,NN);

      myNcoord +=t;

      //   const v3 td = -dfunc * d;
      //   mydev += td;
    }
    std::cout << myNcoord  <<" "<< natA<< std::endl;
    return myNcoord;
  }


};


template<class UnaryFunc>
double parallelAccumulate(const std::vector<PLMD::Vector>& dataIn,
                          const std::vector<unsigned>reaIndexes,
                          UnaryFunc callable, double startingvalue=0.0) {
  unsigned nat=dataIn.size();
  std::vector<v3> wdata(dataIn.size());
  for(auto i=0U; i<dataIn.size(); ++i) {
    wdata[i][0] = dataIn[i][0];
    wdata[i][1] = dataIn[i][1];
    wdata[i][2] = dataIn[i][2];
  }
  float ncoord=startingvalue;

  // #pragma acc data copyin(wdata[0:nat],reaIndexes[0:nat]) \
  //     copyout(derivatives[0:nat],virial[0:9],ncoord)
#pragma acc data copyin(wdata[0:nat],reaIndexes[0:nat],callable) \
        copy(ncoord)
  {
#pragma acc parallel loop gang reduction(+:ncoord)
    for (size_t i = 0; i < nat; i++) {
      ncoord += callable(wdata[reaIndexes[i]],reaIndexes[i],wdata.data(),reaIndexes.data());
    }

  }
  startingvalue = ncoord;
  return startingvalue;
}

}
#endif // myACCinline_hxx
