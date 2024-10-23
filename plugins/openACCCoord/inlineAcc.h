#ifndef plumed_parallel
#define plumed_parallel
#include <cmath>
#include <iostream>
#include <utility>

#include "plumed/tools/Tools.h"
#include "plumed/tools/AtomNumber.h"

#include "Vector.h"
#include "Tensor.h"
#include "Tools_pow.h"


namespace PLMD {

namespace parallel {

template<class UnaryFunc>
double accumulate_sumOP(const std::vector<PLMD::Vector>& dataIn,
                        const std::vector<unsigned>reaIndexes,
                        UnaryFunc callable, double startingvalue=0.0) {

  using v3f= PLMD::wFloat::Vector<float>;
  using tens= PLMD::wFloat::Tensor<float>;
  unsigned nat=dataIn.size();
  std::vector<v3f> wdata(dataIn.size());
  for(auto i=0U; i<dataIn.size(); ++i) {
    wdata[i][0] = dataIn[i][0];
    wdata[i][1] = dataIn[i][1];
    wdata[i][2] = dataIn[i][2];
  }
  float ncoord=startingvalue;

  // #pragma acc data copyin(wdata[0:nat],reaIndexes[0:nat]) \
  //     copyout(derivatives[0:nat],virial[0:9],ncoord)
#pragma acc data copyin(wdata[0:nat],reaIndexes[0:nat],callable) \
        copy(ncoord)
  {
#pragma acc parallel loop gang reduction(+:ncoord)
    for (size_t i = 0; i < nat; i++) {
      ncoord += callable(wdata[reaIndexes[i]],reaIndexes[i],wdata.data(),reaIndexes.data());
    }

  }
  startingvalue = ncoord;
  return startingvalue;
}
}// namespace parallel
}//namespace PLMD
#endif // plumed_parallel
