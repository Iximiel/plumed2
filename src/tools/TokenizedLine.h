
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
#ifndef __PLUMED_tools_TokenizedLine_h
#define __PLUMED_tools_TokenizedLine_h
#include "Tools.h"
#include <map>
#include <vector>

namespace PLMD {
///This class abstracts the input line in tokens.
///
///The underlying container accelerates the lookup for the keywords
class TokenizedLine {
public:
  using mapType=std::map<std::string,std::string,std::less<void>>;
private:
  using vectorIt = typename std::vector<std::string>::iterator;
  using const_vectorIt = typename std::vector<std::string>::const_iterator;
  mapType tokens;
public:

  struct presentAndFound {
    bool present;
    bool found;
  };
/// Initializer from vector iterators
  TokenizedLine(vectorIt begin, vectorIt end);
/// Initializer from vector iterators
  TokenizedLine(const_vectorIt begin, const_vectorIt end);
/// Initializer from vector
  TokenizedLine(const std::vector<std::string>);
///return a plain string with the all the current KEY=value combinations it is possible to clear the tokens after that
  std::string convertToString(bool alsoClear);
///returns the list of the keys:
  std::string keyList(std::string_view sep = ", ");
///return a keyword and its argument
  std::string getKeyword(std::string_view key) const;
/// Return the size of the underlying container;
  std::size_t size() const;
/// Returns true if the underlying container is empty
  bool empty() const;
/// Read a value from the tokens and remove it from the list
  template<typename T>
  presentAndFound readAndRemove(std::string_view key,
                                T& value,
                                int rep=-1);
/// Read a list of values from the tokens and remove it from the list
  template<typename T>
  presentAndFound readAndRemoveVector(std::string_view key,
                                      std::vector<T>& value,
                                      int rep=-1);
///return true if the flag is present and removes it from the tokens
  bool readAndRemoveFlag(std::string_view key);
};



template<typename T>
TokenizedLine::presentAndFound TokenizedLine::readAndRemove(std::string_view key,
    T& value,
    const int replica_index ) {
  auto keytext = tokens.find(key);
  bool present = keytext != tokens.end();
  bool found=false;
  if(present) {
    found = Tools::parse(keytext->second, value,replica_index);
    tokens.erase(keytext);
  }
  return {present, found};
}


template<typename T>
TokenizedLine::presentAndFound TokenizedLine::readAndRemoveVector(std::string_view key,
    std::vector<T>& value,
    const int replica_index ) {
  auto keytext = tokens.find(key);
  bool present = keytext != tokens.end();
  bool found=false;
  if(present) {
    found = Tools::parseVector(keytext->second, value,replica_index);
    tokens.erase(keytext);
  }
  return {present, found};
}


} //namespace PLMD
#endif //__PLUMED_tools_TokenizedLine_h

