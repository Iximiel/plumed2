#include <cmath>
#include <iostream>
#include <utility>

#include "fastCoord.hxx"
#include "Tools_pow.h"
#include "plumed/tools/Tools.h"
//Tools is a class, so I cannot do:
//using PLMD::Tools::fastpow;

#define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'
// #define plotsize(name) std::cerr << __LINE__ << #name ": "<< name.sizes()<< '\n';

// #define vdbg(...)
namespace myACC {

template <int POW,typename T>
static inline std::pair<T,T> doReducedRational(const T rdist, T result=0.0) {
  const T rNdist=PLMD::Tools::fastpow<POW-1>(rdist);
  result=1.0/(1.0+rNdist*rdist);
  const T dfunc = -rNdist*result*result*POW;
  return {result,dfunc};
}

template <typename T>
static inline std::pair<T,T> doReducedRational(const T rdist, const unsigned pow, T result=0.0) {
  const T rNdist=myACC::Tools::fastpow(rdist, pow-1);
  result=1.0/(1.0+rNdist*rdist);
  const T dfunc = -rNdist*result*result*pow;
  return {result,dfunc};
}

template <int N,typename T>
struct calculatorReducedRationalFixed {
  static std::pair<T,T> calculateSqr(const T distance2,
                                     const T invr0_2,
                                     const T dmax_2,
                                     const T stretch,
                                     const T shift,
                                     const unsigned ) {
    if (distance2 > dmax_2) {
      return {0.0f,0.0f};
    }
    const T rdist = distance2*invr0_2;
    auto [result,dfunc] = doReducedRational<N/2>(rdist);
    dfunc*=2*invr0_2;
    // stretch:
    result=result*stretch+shift;
    dfunc*=stretch;
    return {result,dfunc};
  }
};

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
  float s0=calculator::calculateSqr(0.0f,invr0_2,dmaxsq+1.0,1.0,0.0,pow).first;
  float sd=calculator::calculateSqr(dmaxsq,invr0_2,dmaxsq+1.0,1.0,0.0,pow).first;

  float const stretch=1.0f/(s0-sd);
  float const shift=-sd*stretch;
  return {shift,stretch};
}

using mycalculator = calculatorReducedRationalFlexible<float>;
// using mycalculator = calculatorReducedRationalFixed<6,float>;

fastCoord::fastCoord(unsigned const natA_,
                     unsigned const natB_,
                     unsigned const N,
                     unsigned const M,
                     float const invr0_,
                     float const dmax_)
  : natA(natA_),
    natB(natB_),
    NN(N),
    invr0_2(invr0_*invr0_),
    dmaxsq(dmax_*dmax_) {
  auto [setShift, setStretch] = getShiftAndStretch<mycalculator>(invr0_2, dmaxsq,NN);
  shift = setShift;
  stretch = setStretch;
}

std::pair<float,PLMD::wFloat::Tensor<float>> fastCoord::operator()(
      const PLMD::wFloat::Vector<float>* const positions,
      const PLMD::AtomNumber* const reaIndexes,
      PLMD::wFloat::Vector<float>* const derivatives
) const {
  using v3= PLMD::wFloat::Vector<float>;
  using tens= PLMD::wFloat::Tensor<float>;
  //const T* is a pointer to a constant variable
  //T* const is a constant pointer to a variable
  //const T* const is a constant pointer to a constant variable
  // |                   | *x=something; | x=pointer; |
  // | const T* x;       |    can't      |    can     |
  // | T* const x;       |    can        |    can't   |
  // | const T* const x; |    can't      |    can't   |

  const unsigned nat = natA+natB;
  float ncoord;
  tens outVirial;
  //openacc does not do the reduction on the operator of the Tensor
  //so we are tricking it into doing it anyway ;)
  float* virial = outVirial.data();
#pragma acc data copyin(positions[0:nat],reaIndexes[0:nat]) \
        copyout(derivatives[0:nat],virial[0:9],ncoord)
  {
#pragma acc parallel
    {
      ncoord = 0.0f;
      virial[0] = 0.0f;
      virial[1] = 0.0f;
      virial[2] = 0.0f;
      virial[3] = 0.0f;
      virial[4] = 0.0f;
      virial[5] = 0.0f;
      virial[6] = 0.0f;
      virial[7] = 0.0f;
      virial[8] = 0.0f;
    }

    if (natB==0) {//self
#pragma acc parallel loop gang reduction(+:ncoord,virial[0:9])
      for (size_t i = 0; i < natA; i++) {
        float myNcoord=0.0f;
        v3 mydev= {0.0f, 0.0f, 0.0f};
        tens myVirial;
        v3 xyz=positions[i];
        auto realIndex_i=reaIndexes[i];
//this needs some more work to function correctly
// #pragma acc loop worker reduction(+:myNcoord,mydevX,mydevY,mydevZ, \
//         myVirial_0,myVirial_1,myVirial_2, \
//         myVirial_3,myVirial_4,myVirial_5, \
//         myVirial_6,myVirial_7,myVirial_8)
#pragma acc loop seq
        for (size_t j = 0; j < natA; j++) {
          if(realIndex_i==reaIndexes[j]) {
            continue;
          }
          const v3 d=positions[j]-xyz;
          const float dsq=d.modulo2();
          const auto [t,dfunc ]=mycalculator::calculateSqr(dsq,invr0_2,dmaxsq, stretch,shift,NN);

          myNcoord +=t;

          const v3 td = -dfunc * d;
          mydev += td;
          if(i>j) {
            myVirial+=tens(td,d);
          }
        }
        ncoord += 0.5 * myNcoord;
        //openacc does not do the reduction on the operator of the Tensor
        virial[0]+=myVirial[0][0];
        virial[1]+=myVirial[0][1];
        virial[2]+=myVirial[0][2];
        virial[3]+=myVirial[1][0];
        virial[4]+=myVirial[1][1];
        virial[5]+=myVirial[1][2];
        virial[6]+=myVirial[2][0];
        virial[7]+=myVirial[2][1];
        virial[8]+=myVirial[2][2];
        derivatives[i] = mydev;
      }
    } else {
#pragma acc parallel loop gang reduction(+:ncoord,virial[0:9])
      for (size_t i = 0; i < natA; i++) {
        float myNcoord=0.0f;
        v3 mydev= {0.0f, 0.0f, 0.0f};
        tens myVirial;
        v3 xyz=positions[i];
        auto realIndex_i=reaIndexes[i];
//this needs some more work to function correctly
// #pragma acc loop worker reduction(+:myNcoord,mydevX,mydevY,mydevZ, \
//         myVirial_0,myVirial_1,myVirial_2, \
//         myVirial_3,myVirial_4,myVirial_5, \
//         myVirial_6,myVirial_7,myVirial_8)
#pragma acc loop seq
        for (size_t j = natA; j < nat; j++) {
          if(realIndex_i==reaIndexes[j]) {
            continue;
          }
          const v3 d=positions[j]-xyz;

          const auto [t,dfunc ]=mycalculator::calculateSqr(d.modulo2(),invr0_2,dmaxsq, stretch,shift,NN);

          myNcoord +=t;

          const v3 td = -dfunc * d;
          mydev += td;

          myVirial+=tens(td,d);

        }
        ncoord += myNcoord;
        virial[0]+=myVirial[0][0];
        virial[1]+=myVirial[0][1];
        virial[2]+=myVirial[0][2];
        virial[3]+=myVirial[1][0];
        virial[4]+=myVirial[1][1];
        virial[5]+=myVirial[1][2];
        virial[6]+=myVirial[2][0];
        virial[7]+=myVirial[2][1];
        virial[8]+=myVirial[2][2];
        derivatives[i] = mydev;
      }
//second loop to extract the derivatives of the second group
#pragma acc parallel loop gang
      for (size_t i = natA; i < nat; i++) {
        v3 mydev= {0.0f, 0.0f, 0.0f};
        v3 xyz=positions[i];
        auto realIndex_i=reaIndexes[i];
//this needs some more work to function correctly
// #pragma acc loop worker reduction(+:myNcoord,mydevX,mydevY,mydevZ, \
//         myVirial_0,myVirial_1,myVirial_2, \
//         myVirial_3,myVirial_4,myVirial_5, \
//         myVirial_6,myVirial_7,myVirial_8)
#pragma acc loop seq
        for (size_t j = 0; j < natA; j++) {
          if(realIndex_i==reaIndexes[j]) {
            continue;
          }
          const v3 d=positions[j]-xyz;
          const auto [t,dfunc ]=mycalculator::calculateSqr(d.modulo2(),invr0_2,dmaxsq, stretch,shift,NN);

          mydev -= dfunc*d;
        }
        derivatives[i] = mydev;
      }
    }

  } //data clause
  return {ncoord,outVirial};
}

} // namespace myAcc
