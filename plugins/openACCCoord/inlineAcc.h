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
#ifndef plumed_parallel
#define plumed_parallel
#include <iostream>
#include <utility>

#define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'


#include "Vector.h"
#include "Tensor.h"
#include "Tools_pow.h"


namespace PLMD {

namespace parallel {

template<typename T, typename I,typename S, class Callable>
double accumulate_sumOP(const std::vector<T>& dataIn,
                        const std::vector<I>& reaIndexes,
                        std::vector<T>& dataOut,
                        Tensor & virialOut,
                        const S support,
                        Callable func,
                        double startingvalue=0.0) {

  using v3f = PLMD::wFloat::Vector<float>;
  using tensf= PLMD::wFloat::Tensor<float>;
  unsigned nat=dataIn.size();
  std::vector<v3f> wdata(dataIn.size());
  for(auto i=0U; i<dataIn.size(); ++i) {
    wdata[i][0] = dataIn[i][0];
    wdata[i][1] = dataIn[i][1];
    wdata[i][2] = dataIn[i][2];
  }
  std::vector<v3f> odata(dataOut.size());


  tensf virialTMP;

  float wval=startingvalue;
  {
    float* virial = virialTMP.data();
#pragma acc data copyin(wdata[0:nat],reaIndexes[0:nat],support,func) \
        copy(wval,virial) \
        copyout(odata[0:nat])
    {
      std::array<float,9> myVirial;
#pragma acc parallel loop gang reduction(+:wval,virial[0:9])
      for (size_t i = 0; i < nat; i++) {

        auto [val,dval] = func(i,wdata,reaIndexes,myVirial,support);
        odata[i] = dval;

        wval += val;
        virial[0]+=myVirial[0];
        virial[1]+=myVirial[1];
        virial[2]+=myVirial[2];
        virial[3]+=myVirial[3];
        virial[4]+=myVirial[4];
        virial[5]+=myVirial[5];
        virial[6]+=myVirial[6];
        virial[7]+=myVirial[7];
        virial[8]+=myVirial[8];
      }

    }
  }
  startingvalue = wval;
  std::copy(virialTMP.data(),virialTMP.data()+9,&virialOut[0][0]);
  for(auto i=0U; i<dataOut.size(); ++i) {
    dataOut[i][0] = odata[i][0];
    dataOut[i][1] = odata[i][1];
    dataOut[i][2] = odata[i][2];
  }
  return startingvalue;
}

template<typename T, typename I,typename S, class Callable>
double accumulate_sumOP(const std::vector<T>& dataIn,
                        const std::vector<I>& reaIndexes,
                        std::vector<T>& dataOut,
                        const S support,
                        Callable func,
                        double startingvalue=0.0) {

  using v3f = PLMD::wFloat::Vector<float>;

  unsigned nat=dataIn.size();
  std::vector<v3f> wdata(dataIn.size());
  for(auto i=0U; i<dataIn.size(); ++i) {
    wdata[i][0] = dataIn[i][0];
    wdata[i][1] = dataIn[i][1];
    wdata[i][2] = dataIn[i][2];
  }
  std::vector<v3f> odata(dataOut.size());
  float wval=startingvalue;
#pragma acc data copyin(wdata[0:nat],reaIndexes[0:nat],support,func) \
        copy(wval) \
        copyout(odata[0:nat])
  {
#pragma acc parallel loop gang reduction(+:wval)
    for (size_t i = 0; i < nat; i++) {
      auto [val,dval] = func(i,wdata,reaIndexes,support);
      wval += val;
      odata[i] = dval;
    }

  }
  startingvalue = wval;
  for(auto i=0U; i<dataOut.size(); ++i) {
    dataOut[i][0] = odata[i][0];
    dataOut[i][1] = odata[i][1];
    dataOut[i][2] = odata[i][2];
  }
  return startingvalue;
}

template<typename T, typename I,typename S, class Callable>
double accumulate_sumOP(const std::vector<T>& dataIn,
                        const std::vector<I>& reaIndexes,
                        const S support,
                        Callable func,
                        double startingvalue=0.0) {

  using v3f = PLMD::wFloat::Vector<float>;

  unsigned nat=dataIn.size();
  std::vector<v3f> wdata(dataIn.size());
  for(auto i=0U; i<dataIn.size(); ++i) {
    wdata[i][0] = dataIn[i][0];
    wdata[i][1] = dataIn[i][1];
    wdata[i][2] = dataIn[i][2];
  }
  float wval=startingvalue;
#pragma acc data copyin(wdata[0:nat],reaIndexes[0:nat],support,func) \
        copy(wval)
  {
#pragma acc parallel loop gang reduction(+:wval)
    for (size_t i = 0; i < nat; i++) {
      wval += func(i,wdata,reaIndexes,support);
    }

  }
  startingvalue = wval;
  return startingvalue;
}

template<typename T, typename I, class Callable>
double accumulate_sumOP(const std::vector<T>& dataIn,
                        const std::vector<I>& reaIndexes,
                        Callable func,
                        double startingvalue=0.0) {

  using v3f = PLMD::wFloat::Vector<float>;

  unsigned nat=dataIn.size();
  std::vector<v3f> wdata(dataIn.size());
  for(auto i=0U; i<dataIn.size(); ++i) {
    wdata[i][0] = dataIn[i][0];
    wdata[i][1] = dataIn[i][1];
    wdata[i][2] = dataIn[i][2];
  }
  float wval=startingvalue;

#pragma acc data copyin(wdata[0:nat],reaIndexes[0:nat],func) \
        copy(wval)
  {
#pragma acc parallel loop gang reduction(+:wval)
    for (size_t i = 0; i < nat; i++) {
      wval += func(i,wdata,reaIndexes);
    }

  }
  startingvalue = wval;
  return startingvalue;
}


template<typename T, typename I, class Callable>
double accumulate_sumOP(const std::vector<T>& dataIn,
                        Callable func,
                        double startingvalue=0.0) {

  using v3f = PLMD::wFloat::Vector<float>;

  unsigned nat=dataIn.size();
  std::vector<v3f> wdata(dataIn.size());
  for(auto i=0U; i<dataIn.size(); ++i) {
    wdata[i][0] = dataIn[i][0];
    wdata[i][1] = dataIn[i][1];
    wdata[i][2] = dataIn[i][2];
  }
  float wval=startingvalue;

#pragma acc data copyin(wdata[0:nat],func) \
        copy(wval)
  {
#pragma acc parallel loop gang reduction(+:wval)
    for (size_t i = 0; i < nat; i++) {
      wval += func(i,wdata);
    }

  }
  startingvalue = wval;
  return startingvalue;
}

}// namespace parallel
}//namespace PLMD
#endif // plumed_parallel
