#include "plumed/tools/Tools.h"
#include <fstream>
#include <iostream>

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

int main() {
//NOTE:: We are simply checking that gw has the expected behaviour
  tee out("output");
  for (auto test : {
         "{abcd},{a b d}","a,b,c,d","{1,2,3},4,5,{6}",
         "@ndx:{file1 second}","@ndx:file,@ndx:{file 1 second}"
       }) {
    out << std::quoted(test)<< '\n';
    gch::small_vector<std::string_view> gws ;
    Tools::getWordsSimple(gws,test,", \t\n");
    auto gw  = Tools::getWords(test,", \t\n");
    for(auto x : gws ) {
      out <<"\tgws:"<< std::quoted(x) << "\n";
    }
    for(auto x : gw ) {
      out <<"\tgw :"<< std::quoted(x) << "\n";
    }
    out <<'\n';
  }

  return 0;
}
