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
std::pair<precision,precision> getShiftAndStretch(dataType x) {
  //this does not modify the original x (it is passed by value)
  //but ensurest the correctness of the results :)
  //TODO: test this by
  /*
  dataType x={init};
  auto [stretch,shift]=getShiftAndStretch<switchFunc>(x);
  x.stretch=stretch;
  x.shift=shift;
  auto [stretch_2nd,shift_2nd]=getShiftAndStretch<switchFunc>(x);
  assert(stretch==stretch_2nd);
  assert(shift==shift_2nd);
  //if you remove the following two lines the assertions will fail
  */
  x.stretch=precision(1.0);
  x.shift=precision(0.0);
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
template<typename T>
constexpr T moreThanOne=T(1.0)+T(5.0e10)*myACC::epsilon<T>;
template<typename T>
constexpr T lessThanOne=T(1.0)-T(5.0e10)*myACC::epsilon<T>;
template<>
constexpr float moreThanOne<float> =float(moreThanOne<double>);
template<>
constexpr float lessThanOne<float> =float(lessThanOne<double>);

template <typename T>
static inline std::pair<T,T> doRational(const T rdist,
                                        const unsigned N,
                                        const unsigned M,
                                        const T secDev,
                                        T dfunc,
                                        T result) {
  if(!((rdist > lessThanOne<T>) && (rdist < moreThanOne<T>))) {
    const T rNdist=myACC::Tools::fastpow(rdist,N-1);
    const T rMdist=myACC::Tools::fastpow(rdist,M-1);
    const T num = T(1.0)-rNdist*rdist;
    const T iden = T(1.0)/(T(1.0)-rMdist*rdist);
    result = num*iden;
    dfunc = ((M*result*rMdist)-(N*rNdist))*iden;
  } else {
    //here I imply that the correct initialized value for result and dfunc are being passed
    const T x =(rdist-T(1.0));
    result = result+ x * ( dfunc + 0.5 * x * secDev);
    dfunc  = dfunc + x * secDev;
  }
  return {result,dfunc};
}

template <typename T>
struct calculatorReducedRational {

  static std::pair<T,T> calculate(const T distance,const switchData<T> params) {
    return calculateSqr(distance*distance,params);
  }

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

template<typename T>
struct rationalData {
  using precision = T;
  unsigned natA{0};
  unsigned natB{0};
  unsigned NN{0};
  unsigned MM{0};
  T invr0{1.0};
  T d0{0.0};
  T dmax{1.0};
  T dmaxsq{1.0};
  T stretch{1.0};
  T shift{0.0};
  T centerVal{0.0};
  T secondDev{0.0};
  T centerDev{0.0};
  rationalData() = default;

  rationalData(unsigned const natA_,
               unsigned const natB_,
               int const N,
               int const M,
               T const invr0_,
               T const d0_,
               T const dmax_)
    : natA(natA_),
      natB(natB_),
      NN(N),
      MM(M),
      invr0(invr0_),
      d0(d0_),
      dmax(dmax_),
      dmaxsq(dmax*dmax),
      centerVal(static_cast<T>(NN)/MM),
      secondDev(((N * (M * M - 3.0* M * (-1 + N ) + N *(-3 + 2* N )))/(6.0* M ))),
      centerDev(0.5*NN*(NN-MM)/static_cast<T>(MM)) {}

};

template <typename T>
struct calculatorRational
{
  static std::pair<T,T> calculate(const T distance,const rationalData<T> params) {
    return calculateSqr(distance*distance,params);
  }
  static std::pair<T,T> calculateSqr(const T distance2,const rationalData<T> params) {
    // int mul = (distance2 <= dmax_2);
    if (distance2 > params.dmaxsq) {
      return {0.0f,0.0f};
    }
    auto distance=std::sqrt(distance2);
    const T rdist = (distance-params.d0)*params.invr0;
    T result=1.0;
    T dfunc=0.0;
    if (rdist>0.0) {
      std::tie(result,dfunc) = doRational(rdist, params.NN,
                                          params.MM,params.secondDev,params.centerDev,params.centerVal);
      dfunc*=params.invr0;
      dfunc/=distance;
    }
    // stretch:
    result=result*params.stretch+params.shift;
    dfunc*=params.stretch;
    return {result,dfunc};
  }
};

}
#endif // myACCinline_hxx
