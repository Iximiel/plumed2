/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2023 Daniele Rapetti

The pycv module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The pycv module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "PythonPlumedBase.h"

#include "colvar/Colvar.h"

namespace PLMD {

class NeighborList;

namespace pycv {

///TODO: manual "you have to specify ATOMS=something for default atoms"
///TODO: add interface to pbc
///TODO: the topology can be assumed fixed and done on the go at each run by loading the pdb in the python code
class PythonCVInterface : public Colvar {
  static constexpr auto PYCV_NOTIMPLEMENTED="PYCV_NOTIMPLEMENTED";
  static constexpr auto PYCV_DEFAULTINIT="plumedInit";
  static constexpr auto PYCV_DEFAULTCALCULATE="plumedCalculate";
  static constexpr std::string_view PYCV_COMPONENTPREFIX="py";

  std::unique_ptr<NeighborList> nl{nullptr};
  //the guard MUST be set up before the python objects
  PlumedScopedPythonInterpreter interpreterGuard;
  ::pybind11::module_ py_module {};
  ::pybind11::object py_fcn{};
  ::pybind11::object pyPrepare;
  ::pybind11::object pyUpdate;

  bool pbc=false;
  bool hasPrepare = false;
  bool hasUpdate = false;
  bool invalidateList = true;
  bool firsttime = true;
  void calculateSingleComponent(pybind11::object &);
  void calculateMultiComponent(pybind11::object &);
  void readReturn(const pybind11::object &, Value* );
  void initializeValue(pybind11::dict &);
  void initializeComponent(const std::string&,pybind11::dict &);
  void valueSettings( pybind11::dict &r, Value* valPtr);
public:
//Wraps parse to avoid duplicated kewords
  template<typename T>
  void pyParse(
    const char* key, const ::pybind11::dict &initDict, T& returnValue);
  ::pybind11::dict dataContainer {};
  explicit PythonCVInterface(const ActionOptions&);
// active methods:
  void calculate() override;
  void prepare() override;
  void update() override;

  NeighborList& getNL();
  static void registerKeywords( Keywords& keys );
};

template<typename T>
void PythonCVInterface::pyParse(
  const char* key, const ::pybind11::dict &initDict, T& returnValue) {
  T initVal(returnValue);
  parse(key,returnValue);
  //this is not robust, but with no access to Action::line we cannot use Tools::findKeyword
  if(initDict.contains(key)) {
    if (returnValue != initVal) {
      error(std::string("you specified the same keyword ").append(key)+ " both in python and in the settings file");
    }
    returnValue = initDict[key].cast<T>();
  }
}

} // namespace pycv
} // namespace PLMD
