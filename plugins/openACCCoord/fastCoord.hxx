#ifndef myACC_hxx
#define myACC_hxx
namespace myACC {
class fastCoord {
  unsigned natA{0};
  unsigned natB{0};
  float invr0_2{1.0f};
  float dmaxsq{1.0f};
  float shift{0.0f};
  float stretch{1.0f};
public:
  fastCoord();
  fastCoord(unsigned natA, unsigned natB, float invr0_2, float dmax);
  float operator()(const float* const positions, float* derivatives, float* virial) const;
};

} // namespace myAcc
#endif //myACC_hxx
