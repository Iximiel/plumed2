#include "plumed/tools/AtomNumber.h"
#include "plumed/tools/Communicator.h"
#include "plumed/tools/NeighborList.h"
#include "plumed/tools/Pbc.h"
#include <fstream>
#include <iostream>

using PLMD::AtomNumber;
using PLMD::Communicator;
using PLMD::NeighborList;
using PLMD::Pbc;

// Testing that the Neigbour list will be intialized with the desired number of
// couples

#define check(arg) (((arg)) ? "pass\n" : "not pass\n")

int main(int, char **) {
  std::ofstream report("unitTest");
  Pbc pbc{};
  pbc.setBox(PLMD::Tensor({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  Communicator cm{};
  bool serial = true;
  bool do_pbc = false;
  for (size_t nat0 : {100, 500, 1000, 10000}) {
    std::vector<AtomNumber> list0(nat0);
    size_t i = 0;
    for (auto &an : list0) {
      an.setIndex(i);
      ++i;
    }
    {
      report << "Single list:(" << nat0 << " atoms)\n";
      size_t expected = ((nat0 - 1) * nat0) / 2;
      auto nl = NeighborList(list0, serial, do_pbc, pbc, cm);
      // std::cerr << nl.size() << "==" << expected << ": " << std::boolalpha
      //  << (nl.size() == expected) << '\n';
      report << "[" << nat0
             << "]Initial number: " << check(nl.size() == expected);

      report << "\n";
    }
    for (size_t nat1 : {100, 500, 1000, 10000}) {

      std::vector<AtomNumber> list1(nat1);

      i = 0;
      for (auto &an : list1) {
        an.setIndex(i);
        ++i;
      }

      {
        report << "Double list, no pairs:(" << nat0 << ", " << nat1
               << " atoms)\n";
        bool do_pair = false;
        size_t expected = nat1 * nat0;
        auto nl = NeighborList(list0, list1, serial, do_pair, do_pbc, pbc, cm);
        // std::cerr << nl.size() << "==" <<  expected << ": " << std::boolalpha
        //  << (nl.size() == expected) << '\n';
        report << "[" << nat0 << ", " << nat1
               << "]Initial number: " << check(nl.size() == expected);
        report << "\n";
      }

      if (nat1 == nat0) {
        report << "Double list, with pairs:\n";
        bool do_pair = true;
        size_t expected = nat1;
        auto nl = NeighborList(list0, list1, serial, do_pair, do_pbc, pbc, cm);
        // std::cerr << nl.size() << "==" <<  expected << ": " << std::boolalpha
        //  << (nl.size() == expected) << '\n';
        report << "[" << nat0 << ", " << nat1
               << "]Initial number: " << check(nl.size() == expected);
        report << "\n";
      }
    }
  }
}