/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2025 The plumed team
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
#ifndef __PLUMED_tools_View2D_h
#define __PLUMED_tools_View2D_h
#include <limits>
#include <type_traits>

#include "View.h"

namespace PLMD {

/**A not-owning view for accessing array witha  2D interface

The main idea is to have something that works like the mdspan from c++23.

Views are CHEAP to copy (pointer and an integer), so it is better to pass
them as values



@todo  ctors from std::array and from iterators to parallel the span implementatio
*/
template <typename T, std::size_t N= helpers::dynamic_extent, std::size_t M= helpers::dynamic_extent>
class View2D {
  T *ptr_;
  std::size_t sizeN_;
  std::size_t sizeM_;
public:

  //constructor for fixed size View2D
  template <size_t NN = N, size_t MM = M, typename = std::enable_if_t<NN != helpers::dynamic_extent && MM != helpers::dynamic_extent>>
  View2D(T *p) : ptr_(p), sizeN_(N), sizeM_(M) {}

  //constructor for a View2D with known second dimension
  template <size_t MM = M, typename = std::enable_if_t<MM != helpers::dynamic_extent>>
  View2D(T *p, size_t NN) : ptr_(p), sizeN_(NN), sizeM_(M) {}

  //generic constructor, works also for non fixed view (this might change)
  View2D(T *p, size_t NN, size_t MM) : ptr_(p), sizeN_(NN), sizeM_(MM) {}

  ///returns the size of the first dimension
  constexpr size_t size() const {
    return sizeN_;
  }

  ///returns the View to the i-th row
  constexpr View<T, M> operator[](size_t i) {
    return View<T, M>(ptr_ + i * sizeM_,sizeM_);
  }

  ///returns the reference i-th element
  constexpr const View<T, M> operator[](size_t i) const {
    return View<T, M>(ptr_ + i * sizeM_, sizeM_);
  }

  ///return the pointer to the data
  constexpr T* data() const {
    return ptr_;
  }
};

} // namespace PLMD
#endif // __PLUMED_tools_View2D_h