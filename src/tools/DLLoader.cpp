/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2023 The plumed team
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
#include "DLLoader.h"
#include <cstdlib>

#ifdef __PLUMED_HAS_DLOPEN
#include <dlfcn.h>
#endif

namespace PLMD {

bool DLLoader::installed() {
#ifdef __PLUMED_HAS_DLOPEN
  return true;
#else
  return false;
#endif
}


void* DLLoader::load(const std::string&s, const bool useGlobal) {
#ifdef __PLUMED_HAS_DLOPEN
  int flags=RTLD_NOW | RTLD_LOCAL;
  
  if (useGlobal)
    flags=RTLD_NOW|RTLD_GLOBAL;
  
DLpointer p{dlopen(s.c_str(),flags)};
  if(!p) {
    lastError=dlerror();
  } else {
    lastError="";
    handles.push(p);
  }
  return p.get();
#else
  return NULL;
#endif
}

const std::string & DLLoader::error() {
  return lastError;
}

DLLoader::~DLLoader() {
  auto debug=std::getenv("PLUMED_LOAD_DEBUG");
#ifdef __PLUMED_HAS_DLOPEN
  if(debug)
    std::fprintf(stderr,"delete dlloader\n");
  while(!handles.empty()) {
    handles.pop();
  }
  if(debug)
    std::fprintf(stderr,"end delete dlloader\n");
#endif
}

DLLoader::DLLoader() =default;

void DLDeleter ::   operator()(void* dlAddress) {
#ifdef __PLUMED_HAS_DLOPEN
  int ret=dlclose(dlAddress);
  if(ret)
    std::fprintf(stderr,"+++ error reported by dlclose: %s\n",dlerror());
#endif
}

}
