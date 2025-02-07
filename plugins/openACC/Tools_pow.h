#ifndef __PLUMED_tools_powFloat_h
#define __PLUMED_tools_powFloat_h
#include <type_traits>

//PLMD::Tools is actually a class, so cit cannot be expanded,
// an forthe way I am compiling with openACC the better thing may be not include
// things that are not in std
namespace myACC {
template <typename T>
constexpr T epsilon(std::numeric_limits<T>::epsilon());
namespace Tools {
template <typename T>
constexpr inline
T fastpow(T base, unsigned exp) {
  if(exp<0) {
    exp=-exp;
    base=1.0/base;
  }
  T result = 1.0;
#pragma acc loop seq
  while (exp) {
    if (exp & 1) {
      result *= base;
    }
    exp >>= 1;
    base *= base;
  }
  return result;
}
} //namespace Tools
} //namespace myACC
#endif //__PLUMED_tools_powFloat_h