#ifndef myACC_hxx
#define myACC_hxx
namespace myACC {
float calculateSwitch(unsigned natA,
                      unsigned natB,
                      float* const positions,
                      float* derivatives,
                      float* virial,
                      float invr0_2,
                      float dmax
                     );
} // namespace myAcc
#endif //myACC_hxx
