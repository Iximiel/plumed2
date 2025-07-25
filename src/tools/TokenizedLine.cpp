
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
#include "TokenizedLine.h"

#include <string>
#include <algorithm>

namespace PLMD {

template <typename IT>
auto mapCreator(IT k, IT const end) {
  TokenizedLine::mapType toret;
  for (; k!=end; ++k) {
    auto eqpos=k->find('=');
    // We do not want to modify the original line
    std::string key = k->substr(0,eqpos);
    std::transform(key.begin(),key.end(),key.begin(),::toupper);
    if(eqpos != std::string::npos) {
      toret[key]=k->substr(eqpos+1);
    } else {
      //is a flag
      //maybe giving it a special value to confirm that it is indeed a flag?
      toret[key]="";
    }
  }
  return std::move(toret);
}

TokenizedLine::TokenizedLine(vectorIt begin, vectorIt end):
  tokens(mapCreator(begin, end)) {}

TokenizedLine::TokenizedLine(const_vectorIt begin,
                             const_vectorIt end):
  tokens(mapCreator(begin, end)) {}

TokenizedLine::TokenizedLine(const std::vector<std::string> line):
  PLMD::TokenizedLine(line.begin(),line.end()) {}

std::string TokenizedLine::convertToString(bool alsoClear) {
  std::string output;
  for(auto p=tokens.begin(); p!=tokens.end(); ++p) {
    if( (p->second).find(" " )!=std::string::npos ) {
      output += " " + p->first+ "={" + p->second + "}";
    } else {
      output += " "+ p->first+ "=" + p->second;
    }
  }
  if(alsoClear) {
    tokens.clear();
  }
  return output;
}

std::string TokenizedLine::keyList(std::string_view sep ) {
  std::string mylist="";
  int i=0;
  std::string separator = "";
  for(const auto & l:tokens) {
    mylist = mylist + separator + l.first;
    if(i==0) {
      separator = std::string(sep);
      ++i;
    }
  }
  return mylist;
}

std::size_t TokenizedLine::size() const {
  return tokens.size();
}

bool TokenizedLine::empty() const {
  return tokens.empty();
}

std::string TokenizedLine::getKeyword(std::string_view key) const {
  auto keyArg = tokens.find(key);
  if( keyArg != tokens.end()) {
    return std::string(key) +"="+ keyArg->second;
  }
  return "";
}

bool TokenizedLine::readAndRemoveFlag(std::string_view key) {
  auto keytext = tokens.find(key);
  if(keytext != tokens.end()) {
    tokens.erase(keytext);
    return true;
  }
  return false;
}

} // namespace PLMD

