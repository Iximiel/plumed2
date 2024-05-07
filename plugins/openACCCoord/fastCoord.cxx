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
  float ncoord=0.0f;
    
#pragma acc data copy(ncoord) \
        copyin(positions[0:3*nat]) \
        copyout(derivatives[0:3*nat],virial[0:9])
  {
#pragma acc parallel
{
     virial[0] = 0.0f;//0.0f;
     virial[1] = 0.0f;//1.0f;
     virial[2] = 0.0f;//2.0f;
    virial[3] = 0.0f;//3.0f;
     virial[4] = 0.0f;//4.0f;
     virial[5] = 0.0f;//5.0f;
     virial[6] = 0.0f;//6.0f;
     virial[7] = 0.0f;//7.0f;
     virial[8] = 0.0f;//8.0f;
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

//this needs some more work to functionc correctly
// #pragma acc loop worker reduction(+:myNcoord,mydevX,mydevY,mydevZ, \
//         myVirial_0,myVirial_1,myVirial_2, \
//         myVirial_3,myVirial_4,myVirial_5, \
//         myVirial_6,myVirial_7,myVirial_8)
#pragma acc loop seq
        for (size_t j = 0; j < natA; j++) {

          const float d0=positions[3*j  ]-positions[3*i  ];
          const float d1=positions[3*j+1]-positions[3*i+1];
          const float d2=positions[3*j+2]-positions[3*i+2];

          float dsq=d0*d0+d1*d1+d2*d2;
          //todo this:
          //if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;

          //add will need to be the check on the same "real" id
          bool add=i!=j;
          auto [t,dfunc ]=calculateSqr<6>(dsq,invr0_2,dmaxsq, stretch,shift);
          myNcoord +=add*t;

          // dfunc*=add;

          const float t_0 = -dfunc * d0;
          const float t_1 = -dfunc * d1;
          const float t_2 = -dfunc * d2;
          mydevX += t_0;
          mydevY += t_1;
          mydevZ += t_2;
          add&=i>j;
          myVirial_0 += t_0 * d0 * add;
          myVirial_1 += t_0 * d1 * add;
          myVirial_2 += t_0 * d2 * add;
          myVirial_3 += t_1 * d0 * add;
          myVirial_4 += t_1 * d1 * add;
          myVirial_5 += t_1 * d2 * add;
          myVirial_6 += t_2 * d0 * add;
          myVirial_7 += t_2 * d1 * add;
          myVirial_8 += t_2 * d2 * add;
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

