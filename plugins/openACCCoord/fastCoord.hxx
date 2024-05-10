#ifndef myACC_hxx
#define myACC_hxx
#include "plumed/tools/AtomNumber.h"
namespace myACC {
class fastCoord {
  unsigned natA{0};
  unsigned natB{0};
  unsigned NN{0};
  float invr0_2{1.0f};
  float dmaxsq{1.0f};
  float shift{0.0f};
  float stretch{1.0f};
public:
  fastCoord()=default;
  fastCoord(unsigned natA,
            unsigned natB,
            unsigned NN,
            unsigned MM,
            float invr0,
            float dmax);
  float operator()(const float* const positions,
                   const PLMD::AtomNumber* const reaIndexes,
                   float* derivatives,
                   float* virial) const;
};

} // namespace myAcc
#endif //myACC_hxx
