#include <cmath>
#include <type_traits>
#include <iostream>
#include <utility>

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
} //namespace Tools

template <int POW,typename T>
static inline std::pair<T,T> doRational(const T rdist, T result=0.0) {

  const T rNdist=Tools::fastpow<POW-1>(rdist);
  result=1.0/(1.0+rNdist*rdist);
  const T dfunc = -POW*rNdist*result*result;
  // stretch:
  // result=result*stretch+shift;
  // dfunc*=stretch;
  return {result,dfunc};
}
template <int N,typename T>
std::pair<T,T> calculateSqr(const T distance2,
                            const T invr0_2,
                            const T dmax_2=6.0,
                            const T stretch=1.0,
                            const T shift=0.0) {
  int mul = (distance2 <= dmax_2);
  const T rdist = distance2*invr0_2;
  auto [result,dfunc] = doRational<N/2>(rdist);
  dfunc*=2*invr0_2;
  // stretch:
  result=result*stretch+shift;
  dfunc*=stretch;
  result*=mul;
  dfunc*=mul;
  return {result,dfunc};

}

float calculateSwitch(unsigned const natA,
                      unsigned const natB,
                      float* const positions,
                      float* derivatives,
                      float* virial,
                      float const invr0_2,
                      float const dmax) {

  float const dmaxsq=dmax*dmax;

  float s0=calculateSqr<6,float>(0.0f,invr0_2,dmaxsq+1.0).first;
  float sd=calculateSqr<6,float>(dmaxsq,invr0_2,dmaxsq+1.0).first;

  float const stretch=1.0/(s0-sd);
  float const shift=-sd*stretch;
  const unsigned nat = natA+natB;
  vdbg(nat);
  float ncoord=0.0f;
#pragma acc data copy(ncoord) \
        copyin(positions[:3*nat]) \
        copyout(derivatives[:3*nat])
  {
    if (natB==0) {
#pragma acc parallel loop gang reduction(+:ncoord)
      for (size_t i = 0; i < natA; i++) {
        const unsigned atomA=i;
        float myNcoord=0.0f;

        float mydevX=0.0f;
        float mydevY=0.0f;
        float mydevZ=0.0f;

// #pragma acc loop seq
#pragma acc loop vector reduction(+:myNcoord,mydevX,mydevY,mydevZ)
        for (size_t j = 0; j < natA; j++) {
          const unsigned atomB=j;

          const float d0=positions[3*atomB  ]-positions[3*atomA  ];
          const float d1=positions[3*atomB+1]-positions[3*atomA+1];
          const float d2=positions[3*atomB+2]-positions[3*atomA+2];

          float dsq=d0*d0+d1*d1+d2*d2;
          //todo this:
          //if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;


          //add will need to be the check on the same id
          bool add=i!=j;
          auto [t,dfunc ]=calculateSqr<6>(dsq,invr0_2,dmaxsq, stretch,shift);
          myNcoord +=add*t;

          // dfunc*=add;

          float t_0 = -dfunc * d0;
          float t_1 = -dfunc * d1;
          float t_2 = -dfunc * d2;
          mydevX += t_0;
          mydevY += t_1;
          mydevZ += t_2;
          add&=i>j;
          // myVirial_0 += t_0 * d0*add;
          // myVirial_1 += t_0 * d1*add;
          // myVirial_2 += t_0 * d2*add;
          // myVirial_3 += t_1 * d0*add;
          // myVirial_4 += t_1 * d1*add;
          // myVirial_5 += t_1 * d2*add;
          // myVirial_6 += t_2 * d0*add;
          // myVirial_7 += t_2 * d1*add;
          // myVirial_8 += t_2 * d2*add;
        }
        ncoord += 0.5 * myNcoord;
        derivatives[3*atomA+0] = mydevX;
        derivatives[3*atomA+1] = mydevY;
        derivatives[3*atomA+2] = mydevZ;
      }
    }
  }
  return ncoord;
}

} // namespace myAcc

