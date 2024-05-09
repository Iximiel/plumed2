#include <cmath>
#include <type_traits>
#include <iostream>
#include <utility>

#include "fastCoord.hxx"

#define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'
// #define plotsize(name) std::cerr << __LINE__ << #name ": "<< name.sizes()<< '\n';

// #define vdbg(...)
namespace myACC {
namespace Tools {

template <int exp, typename T, std::enable_if_t< (exp >=0), bool> = true>
inline T fastpow_rec(T const base, T result) {
  if constexpr (exp == 0) {
    return result;
  }
  if constexpr (exp & 1) {
    result *= base;
  }
  return fastpow_rec<(exp>>1),T> (base*base, result);
}

template <int exp, typename T>
inline T fastpow(T const base) {
  if constexpr (exp<0) {
    return  fastpow_rec<-exp,T>(1.0/base,1.0);
  } else {
    return fastpow_rec<exp,T>(base, 1.0);
  }
}

template <typename T>
inline
T fastpow(T base, unsigned exp) {
  T result = 1.0;
  while (exp) {
    if (exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}
} //namespace Tools

template <int POW,typename T>
static inline std::pair<T,T> doReducedRational(const T rdist, T result=0.0) {
  const T rNdist=Tools::fastpow<POW-1>(rdist);
  result=1.0/(1.0+rNdist*rdist);
  const T dfunc = -rNdist*result*result*POW;
  return {result,dfunc};
}

template <typename T>
static inline std::pair<T,T> doReducedRational(const T rdist, const unsigned pow, T result=0.0) {
  const T rNdist=Tools::fastpow(rdist, pow-1);
  result=1.0/(1.0+rNdist*rdist);
  const T dfunc = -rNdist*result*result*pow;
  return {result,dfunc};
}

template <int N,typename T>
struct calculatorFixed {
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
struct calculatorFlexible {
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

using mycalculator = calculatorFlexible<float>;
// using mycalculator = calculatorFixed<6,float>;

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

float fastCoord::operator()(
  const float* const positions,
  const unsigned* const reaIndexes,
  float* const derivatives,
  float* const virial) const {
  //const T* is a pointer to a constant variable
  //T* const is a constant pointer to a variable
  //const T* const is a constant pointer to a constant variable
  // |                   | *x=something; | x=pointer; |
  // | const T* x;       |    can't      |    can     |
  // | T* const x;       |    can        |    can't   |
  // | const T* const x; |    can't      |    can't   |

  const unsigned nat = natA+natB;
  float ncoord;

#pragma acc data copyin(positions[0:3*nat],reaIndexes[0:nat]) \
        copyout(derivatives[0:3*nat],virial[0:9],ncoord)
  {
#pragma acc parallel
    {
      ncoord=0.0f;
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

        float mydevX=0.0f;
        float mydevY=0.0f;
        float mydevZ=0.0f;

        float myVirial_0=0.0f;
        float myVirial_1=0.0f;
        float myVirial_2=0.0f;
        float myVirial_3=0.0f;
        float myVirial_4=0.0f;
        float myVirial_5=0.0f;
        float myVirial_6=0.0f;
        float myVirial_7=0.0f;
        float myVirial_8=0.0f;

        const float x=positions[3*i  ];
        const float y=positions[3*i+1];
        const float z=positions[3*i+2];
        unsigned realIndex_i=reaIndexes[i];
//this needs some more work to functionc correctly
// #pragma acc loop worker reduction(+:myNcoord,mydevX,mydevY,mydevZ, \
//         myVirial_0,myVirial_1,myVirial_2, \
//         myVirial_3,myVirial_4,myVirial_5, \
//         myVirial_6,myVirial_7,myVirial_8)
#pragma acc loop seq
        for (size_t j = 0; j < natA; j++) {

          const float d0=positions[3*j  ]-x;
          const float d1=positions[3*j+1]-y;
          const float d2=positions[3*j+2]-z;

          float dsq=d0*d0+d1*d1+d2*d2;
          //todo this:
          if(realIndex_i==reaIndexes[j]) {
            continue;
          }

          //add will need to be the check on the same "real" id

          // const auto [t,dfunc ]=calculateSqr<6>(dsq,invr0_2,dmaxsq, stretch,shift);
          const auto [t,dfunc ]=mycalculator::calculateSqr(dsq,invr0_2,dmaxsq, stretch,shift,NN);
          myNcoord +=t;

          // dfunc*=add;

          const float t_0 = -dfunc * d0;
          const float t_1 = -dfunc * d1;
          const float t_2 = -dfunc * d2;
          mydevX += t_0;
          mydevY += t_1;
          mydevZ += t_2;
          if(i>j) {
            myVirial_0 += t_0 * d0;
            myVirial_1 += t_0 * d1;
            myVirial_2 += t_0 * d2;
            myVirial_3 += t_1 * d0;
            myVirial_4 += t_1 * d1;
            myVirial_5 += t_1 * d2;
            myVirial_6 += t_2 * d0;
            myVirial_7 += t_2 * d1;
            myVirial_8 += t_2 * d2;
          }
        }
        ncoord += 0.5 * myNcoord;

        virial[0] += myVirial_0;
        virial[1] += myVirial_1;
        virial[2] += myVirial_2;
        virial[3] += myVirial_3;
        virial[4] += myVirial_4;
        virial[5] += myVirial_5;
        virial[6] += myVirial_6;
        virial[7] += myVirial_7;
        virial[8] += myVirial_8;

        derivatives[3*i+0] = mydevX;
        derivatives[3*i+1] = mydevY;
        derivatives[3*i+2] = mydevZ;
      }
    }//self

  } //data clause

  return ncoord;
}

} // namespace myAcc
