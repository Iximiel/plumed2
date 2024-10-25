/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2013-2024 The plumed team
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
#ifndef myACCinline_hxx
#define myACCinline_hxx
#include <cmath>
#include <iostream>
#include <utility>

#include "plumed/tools/AtomNumber.h"

#include "Vector.h"
#include "Tensor.h"
#include "Tools_pow.h"
// #define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'

namespace myACC {

using v3= PLMD::wFloat::Vector<float>;
using tens= PLMD::wFloat::Tensor<float>;

template<typename T>
struct switchData {
  using precision = T;
  unsigned natA{0};
  unsigned natB{0};
  unsigned NN{0};
  T invr0_2{1.0};
  T dmaxsq{1.0};
  T stretch{1.0};
  T shift{0.0};
  switchData() = default;
  // switchData(const switchData&) = default;
  // switchData(switchData&&) = default;
  // switchData& operator=(const switchData&) = default;
  // switchData& operator=(switchData&&) = default;
  switchData(unsigned const natA_,
             unsigned const natB_,
             unsigned const N,
             unsigned const M,
             T const invr0_,
             T const dmax_)
    : natA(natA_),
      natB(natB_),
      NN(N),
      invr0_2(invr0_*invr0_),
      dmaxsq(dmax_*dmax_) {}

};

template <typename switchFunc,
          typename dataType,
          typename precision=typename dataType::precision>
std::pair<precision,precision> getShiftAndStretch(dataType const x) {
  const precision s0=switchFunc::calculateSqr(0.0f,x).first;
  const precision sd=switchFunc::calculateSqr(x.dmaxsq,x).first;

  const precision stretch=1.0f/(s0-sd);
  const precision shift=-sd*stretch;

  return {stretch,shift};
}

template <typename T>
static inline std::pair<T,T> doReducedRational(const T rdist, const unsigned power, T result=0.0) {
  const T rNdist=myACC::Tools::fastpow(rdist, power-1);
  result=1.0/(1.0+rNdist*rdist);
  const T dfunc = -rNdist*result*result*power;
  return {result,dfunc};
}
template <typename T>
struct calculatorReducedRational {
  static std::pair<T,T> calculateSqr(const T distance2,const switchData<T> params) {
    // int mul = (distance2 <= dmax_2);
    if (distance2 > params.dmaxsq) {
      return {0.0f,0.0f};
    }
    const T rdist = distance2*params.invr0_2;
    auto [result,dfunc] = doReducedRational(rdist,params.NN/2);
    dfunc*=2*params.invr0_2;
    // stretch:
    result=result*params.stretch+params.shift;
    dfunc*=params.stretch;
    return {result,dfunc};
  }
};








}
#endif // myACCinline_hxx
