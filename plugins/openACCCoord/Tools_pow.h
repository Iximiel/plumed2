#include <type_traits>

//PLMD::Tools is actually a class, so cit cannot be expanded,
// an forthe way I am compiling with openACC the better thing may be not include
// things that are not in std
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
