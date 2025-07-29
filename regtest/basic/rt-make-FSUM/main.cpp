#include "plumed/tools/Tools.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
using namespace PLMD;

class tee {
//TODO:: put this tee into a common test utilities lib
  std::ofstream ofs;
public:
  tee(std::string filename) : ofs(filename) {}
  template<typename T>
  tee& operator<<(const T& t) {
    ofs<<t;
    std::cout <<t;
    return *this;
  }
};

template<typename T>
using FSUM=PLMD::Tools::FastStringUnorderedMap<T>;

int main() {
  try {
    tee out("output");
    FSUM<int> mymap;
    //insertions
    std::string addr;
    for (int i=0; i<100; i+=10) {
      addr=std::to_string(i);
      mymap[addr]= i;
    }
    for (const auto &x: {
           "90", "0", "10","20","30","40","50","60","70","80"
         } ) {
      auto kk = mymap.find(x);
      plumed_assert(kk != mymap.end())
          <<"\""<<x << "\" should have been inserted";
      out <<std::left<< std::setw(3) <<x << mymap[x] <<"\n";
      auto k = mymap.getKeys().find(x);
      plumed_assert(k != mymap.getKeys().end())
          <<"The key \"" << x << "\" can't be found in the keylist";
      plumed_assert(&k->first[0] == &k->second[0])
          << "the key \""<<x<<"\", in the keymap, do not point to its its value";
      plumed_assert(&kk->first[0] == &k->second[0])
          << "the key \""<<x<<"\" on the map must point to its value";
    }
    //erasing things:
    for (const auto &x: {
           "90", "0", "10","70","50","40","30","60","20","80"
         } ) {
      out << "Deleting \"" <<x <<"\"";
      mymap.erase(x);
      auto kk = mymap.find(x);

      plumed_assert(kk == mymap.end())
          <<"\""<<x << "\" should have been deleted";
      out <<", remaining elements: "<< mymap.size() << "\n";
      auto k = mymap.getKeys().find(x);
      plumed_assert(k == mymap.getKeys().end())
          <<"The key \"" << x << "\" should not be be found in the keylist anymore";
    }
    out << "The map is empty: " << (mymap.empty()?"true":"false") << "\n";
  } catch(PLMD::Exception &e) {
    std::cerr << "Exception:" << e.what() << "\n";
  }
  return 0;
}
