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
#ifndef __PLUMED_tools_TensorFloat_h
#define __PLUMED_tools_TensorFloat_h

#include "plumed/tools/MatrixSquareBracketsAccess.h"
#include "Vector.h"
#include "LoopUnroller.h"
#include "plumed/tools/Exception.h"

#include <array>

namespace PLMD {
namespace wFloat {
/// Small class to contain local utilities.
/// Should not be used outside of the TensorGeneric class.
template <typename T>
class TensorGenericAux {
public:
// local redefinition, just to avoid including lapack.h here
  static void local_dsyevr(const char *jobz, const char *range, const char *uplo, int *n,
                           T *a, int *lda, T *vl, T *vu, int *
                           il, int *iu, T *abstol, int *m, T *w,
                           T *z__, int *ldz, int *isuppz, T *work,
                           int *lwork, int *iwork, int *liwork, int *info);
};

/**
\ingroup TOOLBOX
Class implementing fixed size matrices of Ts

\tparam n The number rows
\tparam m The number columns

This class implements a matrix of Ts with size fixed at
compile time. It is useful for small fixed size objects (e.g.
3x3 tensors) as it does not waste space to store the vector size.
Moreover, as the compiler knows the size, it can be completely
opimized inline.
Most of the loops are explicitly unrolled using PLMD::LoopUnroller class
Matrix elements are initialized to zero by default. Notice that
this means that constructor is a bit slow. This point might change
in future if we find performance issues.
It takes advantage of MatrixSquareBracketsAccess to provide both
() and [] syntax for access.
Several functions are declared as friends even if not necessary so as to
properly appear in Doxygen documentation.

Aliases are defined to simplify common declarations (Tensor, Tensor2d, Tensor3d, Tensor4d).
Also notice that some operations are only available for 3x3 tensors.

Example of usage
\verbatim

#include "Tensor.h"

using namespace PLMD;

int main(){
  Tensor a;
  TensorGeneric<3,2> b;
  TensorGeneric<3,2> c=matmul(a,b);
  return 0;
}

\endverbatim
*/
template <unsigned n,unsigned m, typename T>
class TensorGeneric:
  public MatrixSquareBracketsAccess<TensorGeneric<n,m,T>,T>
{
  std::array<T,n*m> d;
/// Auxiliary private function for constructor
  void auxiliaryConstructor();
/// Auxiliary private function for constructor
  template<typename... Args>
  void auxiliaryConstructor(T first,Args... arg);
public:
/// Constructor accepting n*m T parameters.
/// Can be used as Tensor<2,2>(1.0,2.0,3.0,4.0)
/// In case a wrong number of parameters is given, a static assertion will fail.
  template<typename... Args>
  TensorGeneric(T first,Args... arg);
/// initialize the tensor to zero
  TensorGeneric();
/// initialize a tensor as an external product of two Vector
  TensorGeneric(const VectorGeneric<n,T>&v1,const VectorGeneric<m,T>&v2);
/// set it to zero
  void zero();
/// access element
  T & operator() (unsigned i,unsigned j);
/// access element
  const T & operator() (unsigned i,unsigned j)const;
/// increment
  TensorGeneric& operator +=(const TensorGeneric<n,m,T>& b);
/// decrement
  TensorGeneric& operator -=(const TensorGeneric<n,m,T>& b);
/// multiply
  TensorGeneric& operator *=(T);
/// divide
  TensorGeneric& operator /=(T);
/// return +t
  TensorGeneric operator +()const;
/// return -t
  TensorGeneric operator -()const;
/// set j-th column
  TensorGeneric& setCol(unsigned j,const VectorGeneric<n,T> & c);
/// set i-th row
  TensorGeneric& setRow(unsigned i,const VectorGeneric<m,T> & r);
/// get j-th column
  VectorGeneric<n,T> getCol(unsigned j)const;
/// get i-th row
  VectorGeneric<m,T> getRow(unsigned i)const;
/// return t1+t2
  template<unsigned n_,unsigned m_,typename U>
  friend TensorGeneric<n_,m_,U> operator+(const TensorGeneric<n_,m_,U>&,const TensorGeneric<n_,m_,U>&);
/// return t1+t2
  template<unsigned n_,unsigned m_,typename U>
  friend TensorGeneric<n_,m_,U> operator-(const TensorGeneric<n_,m_,U>&,const TensorGeneric<n_,m_,U>&);
/// scale the tensor by a factor s
  template<unsigned n_,unsigned m_,typename U>
  friend TensorGeneric<n_,m_,U> operator*(T,const TensorGeneric<n_,m_,U>&);
/// scale the tensor by a factor s
  template<unsigned n_,unsigned m_,typename U>
  friend TensorGeneric<n_,m_,U> operator*(const TensorGeneric<n_,m_,U>&,U s);
/// scale the tensor by a factor 1/s
  template<unsigned n_,unsigned m_,typename U>
  friend TensorGeneric<n_,m_,U> operator/(const TensorGeneric<n_,m_,U>&,U s);
/// returns the determinant
  T determinant()const;
/// return an identity tensor
  static TensorGeneric<n,n,T> identity();
/// return the matrix inverse
  TensorGeneric inverse()const;
/// return the transpose matrix
  TensorGeneric<m,n,T> transpose()const;
/// matrix-matrix multiplication
  template<unsigned n_,unsigned m_,unsigned l_,typename U>
  friend TensorGeneric<n_,l_,U> matmul(const TensorGeneric<n_,m_,U>&,const TensorGeneric<m_,l_,U>&);
/// matrix-vector multiplication
  template<unsigned n_,unsigned m_,typename U>
  friend VectorGeneric<n_,T> matmul(const TensorGeneric<n_,m_,U>&,const VectorGeneric<m_,T>&);
/// vector-matrix multiplication
  template<unsigned n_,unsigned m_,typename U>
  friend VectorGeneric<n_,T> matmul(const VectorGeneric<m_,T>&,const TensorGeneric<m_,n_,U>&);
/// vector-vector multiplication (maps to dotProduct)
  template<unsigned n_>
  friend T matmul(const VectorGeneric<n_,T>&,const VectorGeneric<n_,T>&);
/// matrix-matrix-matrix multiplication
  template<unsigned n_,unsigned m_,unsigned l_,unsigned i_,typename U>
  friend TensorGeneric<n_,i_,U> matmul(const TensorGeneric<n_,m_,U>&,const TensorGeneric<m_,l_,U>&,const TensorGeneric<l_,i_,U>&);
/// matrix-matrix-vector multiplication
  template<unsigned n_,unsigned m_,unsigned l_,typename U>
  friend VectorGeneric<n_,T> matmul(const TensorGeneric<n_,m_,U>&,const TensorGeneric<m_,l_,U>&,const VectorGeneric<l_,T>&);
/// vector-matrix-matrix multiplication
  template<unsigned n_,unsigned m_,unsigned l_,typename U>
  friend VectorGeneric<l_,T> matmul(const VectorGeneric<n_,T>&,const TensorGeneric<n_,m_,U>&,const TensorGeneric<m_,l_,U>&);
/// vector-matrix-vector multiplication
  template<unsigned n_,unsigned m_,typename U>
  friend T matmul(const VectorGeneric<n_,T>&,const TensorGeneric<n_,m_,U>&,const VectorGeneric<m_,T>&);
/// returns the determinant of a tensor
  friend T determinant(const TensorGeneric<3,3,T>&);
/// returns the inverse of a tensor (same as inverse())
  friend TensorGeneric<3,3,T> inverse(const TensorGeneric<3,3,T>&);
/// returns the transpose of a tensor (same as transpose())
  template<unsigned n_,unsigned m_,typename U>
  friend TensorGeneric<n_,m_,U> transpose(const TensorGeneric<m_,n_,U>&);
/// returns the transpose of a tensor (same as TensorGeneric(const VectorGeneric&,const VectorGeneric&))
  template<unsigned n_,unsigned m_,typename U>
  friend TensorGeneric<n_,m_,U> extProduct(const VectorGeneric<n,T>&,const VectorGeneric<m,T>&);
  friend TensorGeneric<3,3,T> dcrossDv1(const VectorGeneric<3,T>&,const VectorGeneric<3,T>&);
  friend TensorGeneric<3,3,T> dcrossDv2(const VectorGeneric<3,T>&,const VectorGeneric<3,T>&);
  friend TensorGeneric<3,3,T> VcrossTensor(const VectorGeneric<3,T>&,const TensorGeneric<3,3,T>&);
  friend TensorGeneric<3,3,T> VcrossTensor(const TensorGeneric<3,3,T>&,const VectorGeneric<3,T>&);
/// Derivative of a normalized vector
  friend TensorGeneric<3,3,T> deriNorm(const VectorGeneric<3,T>&,const TensorGeneric<3,3,T>&);
/// << operator.
/// Allows printing tensor `t` with `std::cout<<t;`
  template<unsigned n_,unsigned m_,typename U>
  friend std::ostream & operator<<(std::ostream &os, const TensorGeneric<n_,m_,U>&);
/// Diagonalize tensor.
/// Syntax is the same as Matrix::diagMat.
/// In addition, it is possible to call if with m_ smaller than n_. In this case,
/// only the first (smaller) m_ eigenvalues and eigenvectors are retrieved.
/// If case lapack fails (info!=0) it throws an exception.
/// Notice that tensor is assumed to be symmetric!!!
  template<unsigned n_,unsigned m_,typename U>
  friend void diagMatSym(const TensorGeneric<n_,n_,U>&,VectorGeneric<m_,T>&evals,TensorGeneric<m_,n_,U>&evec);
};

template <unsigned n,unsigned m, typename T>
void TensorGeneric<n,m,T>::auxiliaryConstructor()
{}

template <unsigned n,unsigned m, typename T>
template<typename... Args>
void TensorGeneric<n,m,T>::auxiliaryConstructor(T first,Args... arg)
{
  d[n*m-(sizeof...(Args))-1]=first;
  auxiliaryConstructor(arg...);
}

template <unsigned n,unsigned m, typename T>
template<typename... Args>
TensorGeneric<n,m,T>::TensorGeneric(T first,Args... arg)
{
  static_assert((sizeof...(Args))+1==n*m,"you are trying to initialize a Tensor with the wrong number of arguments");
  auxiliaryConstructor(first,arg...);
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T>::TensorGeneric() {
  LoopUnroller<n*m,T>::_zero(d.data());
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T>::TensorGeneric(const VectorGeneric<n,T>&v1,const VectorGeneric<m,T>&v2) {
  for(unsigned i=0; i<n; i++)for(unsigned j=0; j<m; j++)d[i*m+j]=v1[i]*v2[j];
}

template <unsigned n,unsigned m, typename T>
void TensorGeneric<n,m,T>::zero() {
  LoopUnroller<n*m,T>::_zero(d.data());
}

template <unsigned n,unsigned m, typename T>
T & TensorGeneric<n,m,T>::operator() (unsigned i,unsigned j) {
#ifdef _GLIBCXX_DEBUG
// index i is implicitly checked by the std::array class
  plumed_assert(j<m);
#endif
  return d[m*i+j];
}

template <unsigned n,unsigned m, typename T>
const T & TensorGeneric<n,m,T>::operator() (unsigned i,unsigned j)const {
#ifdef _GLIBCXX_DEBUG
// index i is implicitly checked by the std::array class
  plumed_assert(j<m);
#endif
  return d[m*i+j];
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T>& TensorGeneric<n,m,T>::operator +=(const TensorGeneric<n,m,T>& b) {
  LoopUnroller<n*m,T>::_add(d.data(),b.d.data());
  return *this;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T>& TensorGeneric<n,m,T>::operator -=(const TensorGeneric<n,m,T>& b) {
  LoopUnroller<n*m,T>::_sub(d.data(),b.d.data());
  return *this;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T>& TensorGeneric<n,m,T>::operator *=(T s) {
  LoopUnroller<n*m,T>::_mul(d.data(),s);
  return *this;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T>& TensorGeneric<n,m,T>::operator /=(T s) {
  LoopUnroller<n*m,T>::_mul(d.data(),1.0/s);
  return *this;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> TensorGeneric<n,m,T>::operator+()const {
  return *this;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> TensorGeneric<n,m,T>::operator-()const {
  TensorGeneric<n,m,T> r;
  LoopUnroller<n*m,T>::_neg(r.d.data(),d.data());
  return r;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T>& TensorGeneric<n,m,T>::setCol(unsigned j,const VectorGeneric<n,T> & c) {
  for(unsigned i=0; i<n; ++i) (*this)(i,j)=c(i);
  return *this;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T>& TensorGeneric<n,m,T>::setRow(unsigned i,const VectorGeneric<m,T> & r) {
  for(unsigned j=0; j<m; ++j) (*this)(i,j)=r(j);
  return *this;
}

template <unsigned n,unsigned m, typename T>
VectorGeneric<n,T> TensorGeneric<n,m,T>::getCol(unsigned j)const {
  VectorGeneric<n,T> v;
  for(unsigned i=0; i<n; ++i) v(i)=(*this)(i,j);
  return v;
}

template <unsigned n,unsigned m, typename T>
VectorGeneric<m,T> TensorGeneric<n,m,T>::getRow(unsigned i)const {
  VectorGeneric<m,T> v;
  for(unsigned j=0; j<m; ++j) v(j)=(*this)(i,j);
  return v;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> operator+(const TensorGeneric<n,m,T>&t1,const TensorGeneric<n,m,T>&t2) {
  TensorGeneric<n,m,T> t(t1);
  t+=t2;
  return t;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> operator-(const TensorGeneric<n,m,T>&t1,const TensorGeneric<n,m,T>&t2) {
  TensorGeneric<n,m,T> t(t1);
  t-=t2;
  return t;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> operator*(const TensorGeneric<n,m,T>&t1,T s) {
  TensorGeneric<n,m,T> t(t1);
  t*=s;
  return t;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> operator*(T s,const TensorGeneric<n,m,T>&t1) {
  return t1*s;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> operator/(const TensorGeneric<n,m,T>&t1,T s) {
  return t1*(1.0/s);
}

template<unsigned n,unsigned m, typename T>
inline
T TensorGeneric<n,m,T>::determinant()const {
static_assert(n==3&&m==3,"determinanat can be called only for 3x3 Tensors");
  return
    d[0]*d[4]*d[8]
    + d[1]*d[5]*d[6]
    + d[2]*d[3]*d[7]
    - d[0]*d[5]*d[7]
    - d[1]*d[3]*d[8]
    - d[2]*d[4]*d[6];
}

//consider to make this a constexpr function
template <unsigned n,unsigned m, typename T>
inline
TensorGeneric<n,n,T> TensorGeneric<n,m,T>::identity() {
  TensorGeneric<n,n,T> t;
  for(unsigned i=0; i<n; i++) t(i,i)=1.0;
  return t;
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<m,n,T> TensorGeneric<n,m,T>::transpose()const {
  TensorGeneric<m,n,T> t;
  for(unsigned i=0; i<m; i++)for(unsigned j=0; j<n; j++) t(i,j)=(*this)(j,i);
  return t;
}

template<unsigned n,unsigned m, typename T>
inline
TensorGeneric<n,m,T> TensorGeneric<n,m,T>::inverse()const {
  static_assert(n==3&&m=3,"inverse can be called only for 3x3 Tensors");
  TensorGeneric t;
  T invdet=1.0/determinant();
  for(unsigned i=0; i<3; i++) for(unsigned j=0; j<3; j++)
      t(j,i)=invdet*(   (*this)((i+1)%3,(j+1)%3)*(*this)((i+2)%3,(j+2)%3)
                        -(*this)((i+1)%3,(j+2)%3)*(*this)((i+2)%3,(j+1)%3));
  return t;
}

template<unsigned n,unsigned m,unsigned l, typename T>
TensorGeneric<n,l,T> matmul(const TensorGeneric<n,m,T>&a,const TensorGeneric<m,l,T>&b) {
  TensorGeneric<n,l,T> t;
  for(unsigned i=0; i<n; i++) for(unsigned j=0; j<l; j++) for(unsigned k=0; k<m; k++) {
        t(i,j)+=a(i,k)*b(k,j);
      }
  return t;
}

template <unsigned n,unsigned m, typename T>
VectorGeneric<n,T> matmul(const TensorGeneric<n,m,T>&a,const VectorGeneric<m,T>&b) {
  VectorGeneric<n,T> t;
  for(unsigned i=0; i<n; i++) for(unsigned j=0; j<m; j++) t(i)+=a(i,j)*b(j);
  return t;
}

template <unsigned n,unsigned m, typename T>
VectorGeneric<n,T> matmul(const VectorGeneric<m,T>&a,const TensorGeneric<m,n,T>&b) {
  VectorGeneric<n,T> t;
  for(unsigned i=0; i<n; i++) for(unsigned j=0; j<m; j++) t(i)+=a(j)*b(j,i);
  return t;
}

template<unsigned n_,typename T>
T matmul(const VectorGeneric<n_,T>&a,const VectorGeneric<n_,T>&b) {
  return dotProduct(a,b);
}

template<unsigned n,unsigned m,unsigned l,unsigned i,typename T>
TensorGeneric<n,i,T> matmul(const TensorGeneric<n,m,T>&a,const TensorGeneric<m,l,T>&b,const TensorGeneric<l,i,T>&c) {
  return matmul(matmul(a,b),c);
}

template<unsigned n,unsigned m,unsigned l, typename T>
VectorGeneric<n,T> matmul(const TensorGeneric<n,m,T>&a,const TensorGeneric<m,l,T>&b,const VectorGeneric<l,T>&c) {
  return matmul(matmul(a,b),c);
}

template<unsigned n,unsigned m,unsigned l, typename T>
VectorGeneric<l,T> matmul(const VectorGeneric<n,T>&a,const TensorGeneric<n,m,T>&b,const TensorGeneric<m,l,T>&c) {
  return matmul(matmul(a,b),c);
}

template <unsigned n,unsigned m, typename T>
T matmul(const VectorGeneric<n,T>&a,const TensorGeneric<n,m,T>&b,const VectorGeneric<m,T>&c) {
  return matmul(matmul(a,b),c);
}

template <typename T>
inline
T determinant(const TensorGeneric<3,3,T>&t) {
  return t.determinant();
}

template <typename T>
inline
TensorGeneric<3,3,T> inverse(const TensorGeneric<3,3,T>&t) {
  return t.inverse();
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> transpose(const TensorGeneric<m,n,T>&t) {
  return t.transpose();
}

template <unsigned n,unsigned m, typename T>
TensorGeneric<n,m,T> extProduct(const VectorGeneric<n,T>&v1,const VectorGeneric<m,T>&v2) {
  return TensorGeneric<n,m,T>(v1,v2);
}

template <typename T>
inline
TensorGeneric<3,3,T> dcrossDv1(const VectorGeneric<3,T>&v1,const VectorGeneric<3,T>&v2) {
  (void) v1; // this is to avoid warnings. still the syntax of this function is a bit dummy...
  return TensorGeneric<3,3,T>(
           0.0, v2[2],-v2[1],
           -v2[2],   0.0, v2[0],
           v2[1],-v2[0],   0.0);
}

template <typename T>
inline
TensorGeneric<3,3,T> dcrossDv2(const VectorGeneric<3,T>&v1,const VectorGeneric<3,T>&v2) {
  (void) v2; // this is to avoid warnings. still the syntax of this function is a bit dummy...
  return TensorGeneric<3,3,T>(
           0.0,-v1[2],v1[1],
           v1[2],0.0,-v1[0],
           -v1[1],v1[0],0.0);
}

template <unsigned n,unsigned m, typename T>
std::ostream & operator<<(std::ostream &os, const TensorGeneric<n,m,T>& t) {
  for(unsigned i=0; i<n; i++)for(unsigned j=0; j<m; j++) {
      if(i!=(n-1) || j!=(m-1)) os<<t(i,j)<<" ";
      else os<<t(i,j);
    }
  return os;
}

/// \ingroup TOOLBOX
typedef TensorGeneric<1,1,double> Tensor1d;
/// \ingroup TOOLBOX
typedef TensorGeneric<2,2,double> Tensor2d;
/// \ingroup TOOLBOX
template<typename T=double>
using Tensor3d = TensorGeneric<3,3,T>;
/// \ingroup TOOLBOX
typedef TensorGeneric<4,4,double> Tensor4d;
/// \ingroup TOOLBOX
typedef TensorGeneric<5,5,double> Tensor5d;
/// \ingroup TOOLBOX
template<typename T=double>
using Tensor = Tensor3d<T>;

template <typename T>
inline
TensorGeneric<3,3,T> VcrossTensor(const VectorGeneric<3,T>&v1,const TensorGeneric<3,3,T>&v2) {

  TensorGeneric<3,3,T> t;
  for(unsigned i=0; i<3; i++) {
    t.setRow(i,matmul(dcrossDv2(v1,v1),v2.getRow(i)));
  }
  return t;
}

template <typename T>
inline
TensorGeneric<3,3,T> VcrossTensor(const TensorGeneric<3,3,T>&v2,const VectorGeneric<3,T>&v1) {
  TensorGeneric<3,3,T> t;
  for(unsigned i=0; i<3; i++) {
    t.setRow(i,-matmul(dcrossDv2(v1,v1),v2.getRow(i)));
  }
  return t;
}

template <typename T>
inline
TensorGeneric<3,3,T> deriNorm(const VectorGeneric<3,T>&v1,const TensorGeneric<3,3,T>&v2) {
  // delta(v) = delta(v1/v1.norm) = 1/v1.norm*(delta(v1) - (v.delta(v1))cross v;
  T over_norm = 1./v1.modulo();
  return over_norm*(v2 - over_norm*over_norm*(extProduct(matmul(v2,v1),v1)));
}

template <unsigned n,unsigned m, typename T>
void diagMatSym(const TensorGeneric<n,n,T>&mat,VectorGeneric<m,T>&evals,TensorGeneric<m,n,T>&evec) {
  // some guess number to make sure work is large enough.
  // for correctness it should be >=20. However, it is recommended to be the block size.
  // I put some likely exaggerated number
  constexpr int bs=100;
  // temporary data, on stack so as to avoid allocations
  std::array<int,10*n> iwork;
  std::array<T,(6+bs)*n> work;
  std::array<int,2*m> isup;
  // documentation says that "matrix is destroyed" !!!
  auto mat_copy=mat;
  // documentation says this is size n (even if m<n)
  std::array<T,n> evals_tmp;
  int nn=n;              // dimension of matrix
  T vl=0.0, vu=1.0; // ranges - not used
  int one=1,mm=m;        // minimum and maximum index
  T abstol=0.0;     // tolerance
  int mout=0;            // number of eigenvalues found (same as mm)
  int info=0;            // result
  int liwork=iwork.size();
  int lwork=work.size();
  TensorGenericAux<T>::local_dsyevr("V", (n==m?"A":"I"), "T", &nn, const_cast<T*>(&mat_copy[0][0]), &nn, &vl, &vu, &one, &mm,
                                 &abstol, &mout, &evals_tmp[0], &evec[0][0], &nn,
                                 isup.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
  if(info!=0) plumed_error()<<"Error diagonalizing matrix\n"
                              <<"Matrix:\n"<<mat<<"\n"
                              <<"Info: "<<info<<"\n";
  plumed_assert(mout==m);
  for(unsigned i=0; i<m; i++) evals[i]=evals_tmp[i];
  // This changes eigenvectors so that the first non-null element
  // of each of them is positive
  // We can do it because the phase is arbitrary, and helps making
  // the result reproducible
  for(unsigned i=0; i<m; ++i) {
    unsigned j=0;
    for(j=0; j<n; j++) if(evec(i,j)*evec(i,j)>1e-14) break;
    if(j<n) if(evec(i,j)<0.0) for(j=0; j<n; j++) evec(i,j)*=-1;
  }
}

static_assert(sizeof(Tensor<double>)==9*sizeof(double), "code cannot work if this is not satisfied");
static_assert(sizeof(Tensor<float>)==9*sizeof(float), "code cannot work if this is not satisfied");

}// wFloat
} //PLMD

#endif

