#include "plumed/tools/Stopwatch.h"
#include "plumed/tools/Tensor.h"
#include "plumed/tools/Vector.h"
#include <fstream>
#include <iostream>

using namespace PLMD;

static void escape(void*p) {
  // this evilness comes from here https://youtu.be/nXaxk27zwlk?t=2451
  //enjoy
  asm volatile ("": : "g"(p): "memory");
  //this actually makes the compiler not opimize the thing pointed by p
}


int main() {
  Stopwatch sw;
  {
    auto t = sw.startStop("InitDouble");
    for (int i = 0; i < 10000000; i++) {
      Vector a(1.0, 2.0, 3.0);
      escape(&a[0]);
      // escape(&a);
      // (void)a;
    }
  }

  {
    auto t = sw.startStop("InitWithConversion");
    for (int i = 0; i < 10000000; i++) {
       Vector a(1, 2, 3);
       escape(&a[0]);
      // escape(&a);
      // (void)a;

    }
  }
  std::ofstream ofs("Timings");
  ofs << sw;

  return 0;
}
