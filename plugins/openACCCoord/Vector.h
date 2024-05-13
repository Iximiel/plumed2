/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2023 The plumed team
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
#ifndef __PLUMED_tools_VectorFloat_h
#define __PLUMED_tools_VectorFloat_h

#include <cmath>
#include <iosfwd>
#include <array>
#include "LoopUnroller.h"

namespace PLMD {
namespace wFloat {

/**
\ingroup TOOLBOX
Class implementing fixed size vectors of Ts

\tparam n The number of elements of the vector.

This class implements a vector of Ts with size fixed at
compile time. It is useful for small fixed size objects (e.g.
3d vectors) as it does not waste space to store the vector size.
Moreover, as the compiler knows the size, it can be completely
opimized inline.
All the methods are inlined for better optimization and
all the loops are explicitly unrolled using PLMD::LoopUnroller class.
Vector elements are initialized to zero by default. Notice that
this means that constructor is a bit slow. This point might change
in future if we find performance issues.
Accepts both [] and () syntax for access.
Several functions are declared as friends even if not necessary so as to
properly appear in Doxygen documentation.

Aliases are defined to simplify common declarations (Vector, Vector2d, Vector3d, Vector4d).
Also notice that some operations are only available for 3 dimensional vectors.

Example of usage
\verbatim
#include "Vector.h"

using namespace PLMD;

int main(){
  VectorGeneric<3, T> v1;
  v1[0]=3.0;
// use equivalently () and [] syntax:
  v1(1)=5.0;
// initialize with components
  VectorGeneric<3, T> v2=VectorGeneric<3, T>(1.0,2.0,3.0);
  VectorGeneric<3, T> v3=crossProduct(v1,v2);
  T d=dotProduct(v1,v2);
  v3+=v1;
  v2=v1+2.0*v3;
}
\endverbatim

*/


template<unsigned n, typename T>
class VectorGeneric {
  std::array<T,n> d;
/// Auxiliary private function for constructor
  void auxiliaryConstructor();
/// Auxiliary private function for constructor
  template<typename... Args>
  void auxiliaryConstructor(T first,Args... arg);
public:
/// Constructor accepting n T parameters.
/// Can be used as Vector<3>(1.0,2.0,3.0) or Vector<2>(2.0,3.0).
/// In case a wrong number of parameters is given, a static assertion will fail.
  template<typename... Args>
  VectorGeneric(T first,Args... arg);
/// create it null
  VectorGeneric();
/// set it to zero
  void zero();
/// array-like access [i]
  T & operator[](unsigned i);
/// array-like access [i]
  const T & operator[](unsigned i)const;
/// parenthesis access (i)
  T & operator()(unsigned i);
/// parenthesis access (i)
  const T & operator()(unsigned i)const;
/// increment
  VectorGeneric& operator +=(const VectorGeneric& b);
/// decrement
  VectorGeneric& operator -=(const VectorGeneric& b);
/// multiply
  VectorGeneric& operator *=(T s);
/// divide
  VectorGeneric& operator /=(T s);
/// sign +
  VectorGeneric operator +()const;
/// sign -
  VectorGeneric operator -()const;
/// return v1+v2
  template<unsigned m>
  friend VectorGeneric<m, T> operator+(const VectorGeneric<m, T>&,const VectorGeneric<m, T>&);
/// return v1-v2
  template<unsigned m, typename U>
  friend VectorGeneric<m, U> operator-(VectorGeneric<m, U>,const VectorGeneric<m, U>&);
/// return s*v
  template<unsigned m>
  friend VectorGeneric<m, T> operator*(T,const VectorGeneric<m, T>&);
/// return v*s
  template<unsigned m>
  friend VectorGeneric<m, T> operator*(const VectorGeneric<m, T>&,T);
/// return v/s
  template<unsigned m>
  friend VectorGeneric<m, T> operator/(const VectorGeneric<m, T>&,T);
/// return v2-v1
  template<unsigned m>
  friend VectorGeneric<m, T> delta(const VectorGeneric<m, T>&v1,const VectorGeneric<m, T>&v2);
/// return v1 .scalar. v2
  template<unsigned m>
  friend T dotProduct(const VectorGeneric<m, T>&,const VectorGeneric<m, T>&);
  //this bad boy produces a warning (in fact becasue declrare the crossproduc as a friend for ALL thhe possible combinations of n and T)
/// return v1 .vector. v2
/// Only available for size 3
  friend VectorGeneric<3,T> crossProduct(const VectorGeneric<3, T>&,const VectorGeneric<3, T>&);
/// compute the squared modulo
  T modulo2()const;
/// Compute the modulo.
/// Shortcut for sqrt(v.modulo2())
  T modulo()const;
/// friend version of modulo2 (to simplify some syntax)
  template<unsigned m>
  friend T modulo2(const VectorGeneric<m, T>&);
/// friend version of modulo (to simplify some syntax)
  template<unsigned m>
  friend T modulo(const VectorGeneric<m, T>&);
/// << operator.
/// Allows printing vector `v` with `std::cout<<v;`
  template<unsigned m>
  friend std::ostream & operator<<(std::ostream &os, const VectorGeneric<m, T>&);
};

template<unsigned n, typename T>
void VectorGeneric<n,T>::auxiliaryConstructor()
{}

template<unsigned n, typename T>
template<typename... Args>
void VectorGeneric<n,T>::auxiliaryConstructor(T first,Args... arg)
{
  d[n-(sizeof...(Args))-1]=first;
  auxiliaryConstructor(arg...);
}

template<unsigned n, typename T>
template<typename... Args>
VectorGeneric<n,T>::VectorGeneric(T first,Args... arg)
{
  static_assert((sizeof...(Args))+1==n,"you are trying to initialize a Vector with the wrong number of arguments");
  auxiliaryConstructor(first,arg...);
}

template<unsigned n, typename T>
void VectorGeneric<n,T>::zero() {
  LoopUnroller<n,T>::_zero(d.data());
}

template<unsigned n, typename T>
VectorGeneric<n,T>::VectorGeneric() {
  LoopUnroller<n,T>::_zero(d.data());
}

template<unsigned n, typename T>
T & VectorGeneric<n,T>::operator[](unsigned i) {
  return d[i];
}

template<unsigned n, typename T>
const T & VectorGeneric<n,T>::operator[](unsigned i)const {
  return d[i];
}

template<unsigned n, typename T>
T & VectorGeneric<n,T>::operator()(unsigned i) {
  return d[i];
}

template<unsigned n, typename T>
const T & VectorGeneric<n,T>::operator()(unsigned i)const {
  return d[i];
}

template<unsigned n, typename T>
VectorGeneric<n,T>& VectorGeneric<n,T>::operator +=(const VectorGeneric<n,T>& b) {
  LoopUnroller<n, T>::_add(d.data(),b.d.data());
  return *this;
}

template<unsigned n, typename T>
VectorGeneric<n,T>& VectorGeneric<n,T>::operator -=(const VectorGeneric<n,T>& b) {
  LoopUnroller<n, T>::_sub(d.data(),b.d.data());
  return *this;
}

template<unsigned n, typename T>
VectorGeneric<n,T>& VectorGeneric<n,T>::operator *=(T s) {
  LoopUnroller<n, T>::_mul(d.data(),s);
  return *this;
}

template<unsigned n, typename T>
VectorGeneric<n,T>& VectorGeneric<n,T>::operator /=(T s) {
  LoopUnroller<n, T>::_mul(d.data(),1.0/s);
  return *this;
}

template<unsigned n, typename T>
VectorGeneric<n,T>  VectorGeneric<n,T>::operator +()const {
  return *this;
}

template<unsigned n, typename T>
VectorGeneric<n,T> VectorGeneric<n,T>::operator -()const {
  VectorGeneric<n,T> r;
  LoopUnroller<n, T>::_neg(r.d.data(),d.data());
  return r;
}

template<unsigned n, typename T>
VectorGeneric<n,T> operator+(const VectorGeneric<n,T>&v1,const VectorGeneric<n,T>&v2) {
  VectorGeneric<n,T> v(v1);
  return v+=v2;
}

template<unsigned n, typename T>
VectorGeneric<n,T> operator-(VectorGeneric<n,T>v1,const VectorGeneric<n,T>&v2) {
  return v1-=v2;
}

template<unsigned n, typename T>
VectorGeneric<n,T> operator*(T s,const VectorGeneric<n,T>&v) {
  VectorGeneric<n,T> vv(v);
  return vv*=s;
}

template<unsigned n, typename T>
VectorGeneric<n,T> operator*(const VectorGeneric<n,T>&v,T s) {
  return s*v;
}

template<unsigned n, typename T>
VectorGeneric<n,T> operator/(const VectorGeneric<n,T>&v,T s) {
  return v*(1.0/s);
}

template<unsigned n, typename T>
VectorGeneric<n,T> delta(const VectorGeneric<n,T>&v1,const VectorGeneric<n,T>&v2) {
  return v2-v1;
}

template<unsigned n, typename T>
T VectorGeneric<n,T>::modulo2()const {
  return LoopUnroller<n, T>::_sum2(d.data());
}

template<unsigned n, typename T>
T dotProduct(const VectorGeneric<n,T>& v1,const VectorGeneric<n,T>& v2) {
  return LoopUnroller<n, T>::_dot(v1.d.data(),v2.d.data());
}

template<typename T>
inline
VectorGeneric<3, T> crossProduct(const VectorGeneric<3, T>& v1,const VectorGeneric<3, T>& v2) {
  return VectorGeneric<3, T>(
           v1[1]*v2[2]-v1[2]*v2[1],
           v1[2]*v2[0]-v1[0]*v2[2],
           v1[0]*v2[1]-v1[1]*v2[0]);
}

template<unsigned n, typename T>
T VectorGeneric<n,T>::modulo()const {
  return sqrt(modulo2());
}

template<unsigned n, typename T>
T modulo2(const VectorGeneric<n,T>&v) {
  return v.modulo2();
}

template<unsigned n, typename T>
T modulo(const VectorGeneric<n,T>&v) {
  return v.modulo();
}

template<unsigned n, typename T>
std::ostream & operator<<(std::ostream &os, const VectorGeneric<n,T>& v) {
  for(unsigned i=0; i<n-1; i++) os<<v(i)<<" ";
  os<<v(n-1);
  return os;
}


/// \ingroup TOOLBOX
/// Alias for one dimensional vectors
typedef VectorGeneric<1,double> Vector1d;
/// \ingroup TOOLBOX
/// Alias for two dimensional vectors
typedef VectorGeneric<2,double> Vector2d;
/// \ingroup TOOLBOX
/// Alias for three dimensional vectors
// typedef VectorGeneric<3,double> Vector3d;
template<typename T=double>
using Vector3d = VectorGeneric<3,T>;
/// \ingroup TOOLBOX
/// Alias for four dimensional vectors
typedef VectorGeneric<4,double> Vector4d;
/// \ingroup TOOLBOX
/// Alias for five dimensional vectors
typedef VectorGeneric<5,double> Vector5d;
/// \ingroup TOOLBOX
/// Alias for three dimensional vectors
// typedef Vector3d Vector;
template<typename T=double>
using Vector = Vector3d<T>;


static_assert(sizeof(VectorGeneric<2,double>)==2*sizeof(double), "code cannot work if this is not satisfied");
static_assert(sizeof(VectorGeneric<3,double>)==3*sizeof(double), "code cannot work if this is not satisfied");
static_assert(sizeof(VectorGeneric<4,double>)==4*sizeof(double), "code cannot work if this is not satisfied");

}// wFloat
} //PLMD

#endif

