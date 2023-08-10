/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2023 The plumed team
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
#include "CLTool.h"
#include "core/CLToolRegister.h"
#include "tools/Tools.h"
#include "config/Config.h"
#include <cstdio>
#include <string>

namespace PLMD {
namespace cltools {
//+PLUMEDOC TOOLS config
/*
This tool allows you to obtain information about your plumed intallation

You can specify the information you require using the following command line
arguments

\par Examples

The following command returns 0 if the module core is installed.
\verbatim
plumed config has core
\endverbatim

*/
//+ENDPLUMEDOC

class CLConfig:
  public CLTool {
   public:
  static void registerKeywords( Keywords& keys );
  explicit CLConfig(const CLToolOptions& co );
  int main(FILE* in, FILE*out,Communicator& pc) override;
  std::string description()const override {
    return "provide informations about your plumed configuration";
  }
};

PLUMED_REGISTER_CLTOOL(CLConfig,"config")

void CLConfig::registerKeywords( Keywords& keys ) {
  CLTool::registerKeywords( keys );
  keys.addFlag("-q",false,"don't write anything, just return true of false");
  keys.addFlag("--quiet",false,"don't write anything, just return true of false");
  //keys.addFlag("show",false,"dump a full configuration file");
  keys.add("optional","has", //[word1 [word2]..]
           "check if plumed has the specified features");
  keys.add("optional","module", //[word1 [word2]..]
           "check if plumed has the specified module enabled");
   keys.addFlag("python_bin",false,"write the path to the python bin and return "
               "if plumed has been conmpiled with it");
   keys.addFlag("mpiexec",false,"write the path to the mpiexec bin and return if"
               " plumed has been conmpiled with it");
           
  //keys.addFlag("makefile_conf",false,"dumps the Makefile.conf file");

}

CLConfig::CLConfig(const CLToolOptions& co ):
  CLTool(co)
{
  inputdata=commandline;
}
int CLConfig::main(FILE* in, FILE*out,Communicator& pc) {

  bool quiet, q;
  parseFlag("-q",q);
  parseFlag("--quiet",quiet);
   quiet|=q;
  //parseFlag("-q/--quiet",quiet);
  std::string moduleCheck;
  bool moduleMode=parse("module",moduleCheck);
  std::string featureCheck;
  bool featureMode=parse("has",featureCheck);
  if(moduleMode) {
    switch (config::plumedHasModule(moduleCheck)) {
    case config::presence::always : [[falltrough]];
    case config::presence::on : {
      if (!quiet) {
        std::fprintf(out,"%s on\n",moduleCheck.c_str());
      }
      return 0;
    }
    break;
    case config::presence::off: {
      if (!quiet) {
        std::fprintf(out,"%s off\n",moduleCheck.c_str());
      }
      return 1;
    }
    break;
    case config::presence::notFound : {
      if (!quiet) {
        std::fprintf(out,"%s not found\n",moduleCheck.c_str());
      }
      return 1;
    }
    break;
    }
  }

  if(featureMode) {
    switch (config::plumedHasFeature(featureCheck)) {
    case config::presence::always : [[falltrough]];//this should not happen
    case config::presence::on : {
      if (!quiet) {
        std::fprintf(out,"%s on\n",featureCheck.c_str());
      }
      return 0;
    }
    break;
    case config::presence::off: {
      if (!quiet) {
        std::fprintf(out,"%s off\n",featureCheck.c_str());
      }
      return 1;
    }
    break;
    case config::presence::notFound : {
      if (!quiet) {
        std::fprintf(out,"%s not found\n",featureCheck.c_str());
      }
      return 1;
    }
    break;
    }
  }
  // bool show;
  // parseFlag("show",show);
  // bool makefile_conf;
  // parseFlag("makefile_conf",makefile_conf);

}
} // namespace cltools
} // namespace PLMD