#include "plumed/tools/Vector.h"
#include "plumed/tools/Tensor.h"
#include "plumed/tools/Stopwatch.h"
#include "plumed/tools/View.h"

#include <numeric>
#include <fstream>
#include <array>
#include <vector>
#include <iostream>
#include <cassert>
#include <typeinfo>

using PLMD::View;

#define displaycall(x) #x << " = " << (x)
///A very barebone tee implementation
class tee {
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

void basics(tee& out);
void nonspanlikeInteractions(tee& out);

int main() {
  tee out("output");
  basics(out);
  nonspanlikeInteractions(out);

  return 0;
}

void basics(tee& out) {
  //View should be used as a way to passing data to function o to not owning classes
  std::array<PLMD::Vector,4> a= {PLMD::Vector{1,1,1},
                                 PLMD::Vector{2,2,2},
                                 PLMD::Vector{3,3,3},
                                 PLMD::Vector{4,4,4}
                                };
  View<PLMD::Vector,4> v(a.data());
  out << "Dimension of View<int,4> should be 4: " << displaycall(v.size()) << "\n";
  assert(v.size()==4);
  out << "View must access to the element of the array: \n";
  for (std::size_t i=0; i<a.size(); ++i) {
    out <<displaycall(i)<< ", " <<displaycall(a[i]) << ", " << displaycall(v[i])  << "\n";
    // asserting on the pointers, since == between floats is kinda criminal
    // and moreover is more on the point
    assert(&a[i][0]==&v[i][0]);
    assert(&a[i][1]==&v[i][1]);
    assert(&a[i][2]==&v[i][2]);
  }
  //Do not own, but in principle has write access
  v[2]=PLMD::Vector(5,5,5);
  for (std::size_t i=0; i<a.size(); ++i) {
    out <<displaycall(i)<< ", " <<displaycall(a[i]) << ", " << displaycall(v[i])  << "\n";
    assert(&a[i][0]==&v[i][0]);
    assert(&a[i][1]==&v[i][1]);
    assert(&a[i][2]==&v[i][2]);
  }
  // const view only have read access,
  // const View<int,4> vc(a.data());
  // the following code will not compile:
  // vc[1]=PLMD::Vector(5,5,5); won't compile

  //View can be used also with dynamic memory:
  std::vector<int> b= {9,8,7,6,5};
  //this won't compile for dynamic momory access
  // View<int> v2(b.data()); won't compile
  View<int> v2(b.data(), b.size());
  //using View<int> is a synonym for View<int,PLMD::helpers::dynamic_extent>
  out << "Dimension of View<int> should be " << b.size() << ": " << displaycall(v2.size()) << "\n";
  assert(v2.size()==b.size());
  out << "View must access to the element of the vector: \n";
  for (std::size_t i=0; i<b.size(); ++i) {
    out <<displaycall(i)<< ", " <<displaycall(b[i]) << ", " << displaycall(v2[i])  << "\n";
    assert(b[i]==v2[i]);
  }
  //And the same rules are true for write access and const view

  v2[2]=9;
  for (std::size_t i=0; i<a.size(); ++i) {
    out <<displaycall(i)<< ", " <<displaycall(b[i]) << ", " << displaycall(v2[i])  << "\n";
    assert(b[i]==v2[i]);
  }

  // you can have also a read only view:
  View<const int> v3(b.data(), b.size());
  //using View<int> is a synonym for View<int,PLMD::helpers::dynamic_extent>
  out << "Dimension of View<const int> should be " << b.size() << ": "
      << displaycall(v3.size()) << "\n";
  assert(v3.size()==b.size());
  out << "View must access to the element of the vector: \n";
  for (std::size_t i=0; i<b.size(); ++i) {
    out <<displaycall(i)<< ", " <<displaycall(b[i]) << ", " << displaycall(v3[i])  << "\n";
    assert(b[i]==v3[i]);
  }
  // And of course you can't write to it
  // v3[2]=9; won't compile

  out << "You can acces the pointer with .data():\n";
  out << displaycall(*v3.data())  << " " << displaycall(v3[0]) << "\n";
  out << "and use iterators:\n";
  for (auto it=v3.begin(); it!=v3.end(); ++it) {
    out <<displaycall(*it)  << "\n";
  }
  out << "and with modern range based for:\n";
  for (auto i:v3) {
    out <<displaycall(i)  << "\n";
  }
  out << "and have some fun with stl algorithms:\n";
  out << displaycall(std::accumulate(v3.begin(), v3.end(), 0))  << "\n";
  out << displaycall(std::accumulate(b.begin(),   b.end(), 0))  << "\n";
}

void nonspanlikeInteractions(tee& out) {
  //Here I am using the three way of obtaining a subview from a view:
  // - v_all.subview<3,3>(); when both offset and count are known at comile time (safest in principle)
  // - v_all.subview<3>(6); when only the count is known at compile time, and the offset is known at run time
  // - v_all.subview(5,2); when both offset and count are known only at run time
  std::vector<double> data(24,0.0);
  View<double> v_all(data.data(),data.size());
  out << "Original data:\n";
  for (const auto& x: data)  {
    out << x <<" ";
  }
  out << "\n";
  //there are also some useful functions:
  //A view<T,3> can be combined with a PLMD::Vector<3>:
  PLMD::Vector myv {1,1,1};
  View<double,3> v1 = v_all.subview<3,3>();
  v1+=myv;
  auto v2 = v_all.subview_n<3>(6);
  v2-=myv;
  v_all.subview<10,3>() = 2.0*myv;

  out << "Data after the +={1,1,1} to head+3 and -={1,1,1} to head+6, and assign={2,2,2} to head+10:\n";
  for (const auto& x: data)  {
    out << x <<" ";
  }
  out << "\n";
  auto v3 = v_all.subview(5,2);
  v3*=-2.0;
  //A View can be *= with a double
  out << "Data: after the *=2.0 on head+5 (2):\n";
  for (const auto& x: data)  {
    out << x <<" ";
  }
  out << "\n";
  //Two view <T,3> can be combined with delta to form a PLMD::Vector<3>
  auto result = delta(v2,v1);
  out << displaycall(result) << "\n";
  //only on the output stream, I prefer to not risk the test outcome on name mangle implementation
  std::cout << displaycall( typeid(delta(v2,v1)).name()) << "\n";
}
