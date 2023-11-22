/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2019-2023 of Toni Giorgino

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
#include "PythonCVInterface.h"

#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/NeighborList.h"
#include "tools/Pbc.h"

#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#define vdbg(...)                                                              \
  std::cerr << std::boolalpha<<std::setw(4) << __LINE__ << ":" << std::setw(20)                \
            << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'

namespace py = pybind11;

using std::string;
using std::vector;
using pycvComm_t = double;

namespace PLMD {
namespace pycv {

PLUMED_REGISTER_ACTION(PythonCVInterface, "PYCVINTERFACE")

void PythonCVInterface::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );
  keys.add("atoms","ATOMS","the list of atoms to be passed to the function");
  //NL
  keys.add("atoms","GROUPA","First list of atoms for the neighbourlist");
  keys.add("atoms","GROUPB","Second list of atoms for the neighbourlist (if empty, N*(N-1)/2 pairs in GROUPA are counted)");
  keys.addFlag("PAIR",false,"Pair only 1st element of the 1st group with 1st element in the second, etc");
  keys.addFlag("NLIST",false,"Use a neighbor list to speed up the calculation");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbor list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbor list");
  //python components
  keys.add("hidden","COMPONENTS","if provided, the function will return multiple components, with the names given");
  keys.addOutputComponent(PYCV_COMPONENTPREFIX.data(),"COMPONENTS","Each of the components output py the Python code, prefixed by py-");
  //python calling
  keys.add("compulsory","IMPORT","the python file to import, containing the function");
  keys.add("compulsory","CALCULATE",PYCV_DEFAULTCALCULATE,"the function to call as calculate method of a CV");
  keys.add("compulsory","INIT",PYCV_DEFAULTINIT,"the function to call during the construction method of the CV");
  // python: add other callable methods
  keys.add("compulsory","PREPARE",PYCV_NOTIMPLEMENTED,"the function to call as prepare method of the CV");
  keys.add("compulsory","UPDATE", PYCV_NOTIMPLEMENTED,"the function to call as update() method of the CV");

  // NOPBC is in Colvar!
}

PythonCVInterface::PythonCVInterface(const ActionOptions&ao)try ://the catch only applies to pybind11 things
  PLUMED_COLVAR_INIT(ao),
  interpreterGuard(log) {
  //let's check the python things at first
  std::string import;
  parse("IMPORT",import);
  std::string calculateFunName;
  parse("CALCULATE",calculateFunName);
  log.printf("  will import %s and call function %s\n", import.c_str(),
             calculateFunName.c_str());
  // Initialize the module and function pointers
  py_module = py::module::import(import.c_str());
  if (!py::hasattr(py_module,calculateFunName.c_str())) {
    error("the function " + calculateFunName + " is not present in "+ import);
  }

  py_fcn = py_module.attr(calculateFunName.c_str());
  std::string initFunName;
  parse("INIT",initFunName);
  py::dict initDict;
  if(py::hasattr(py_module,initFunName.c_str())) {
    log.printf("  will use %s during the initialization\n", initFunName.c_str());
    auto initFcn = py_module.attr(initFunName.c_str());
    if (py::isinstance<py::dict>(initFcn)) {
      initDict = initFcn;
    } else {
      initDict = initFcn(this);
    }
  } else if(initFunName!=PYCV_DEFAULTINIT) {
    //If the default INIT is not preset, is not a problem
    error("the function "+ initFunName + " is not present in "+ import);
  }

  std::string prepareFunName;
  parse("PREPARE",prepareFunName);
  if (prepareFunName!=PYCV_NOTIMPLEMENTED) {
    if (!py::hasattr(py_module,prepareFunName.c_str())) {
      error("the function " + prepareFunName + " is not present in "+ import);
    }
    hasPrepare=true;
    pyPrepare=py_module.attr(prepareFunName.c_str());
    log.printf("  will use %s while calling prepare() before calculate()\n", prepareFunName.c_str());
  }

  std::string updateFunName;
  parse("UPDATE",updateFunName);
  if (updateFunName!=PYCV_NOTIMPLEMENTED) {
    if (!py::hasattr(py_module,updateFunName.c_str())) {
      error("the function " + updateFunName + " is not present in " + import);
    }
    pyUpdate=py_module.attr(updateFunName.c_str());
    hasUpdate=true;
    log.printf("  will use %s while calling update() after calculate()\n", updateFunName.c_str());
  }

  {
    std::vector<std::string> components;
    parseVector("COMPONENTS", components);

    if (components.size()>1) {
      error("Please define multiple COMPONENTS from INIT in python.");
    }
  }
  if(initDict.contains("COMPONENTS")) {
    if(initDict.contains("Value")) {
      error("The initialize dict cannot contain both \"Value\" and \"COMPONENTS\"");
    }
    if(!py::isinstance<py::dict>(initDict["COMPONENTS"])) {
      error("COMPONENTS must be a dictionary using with the name of the components as keys");
    }
    py::dict components=initDict["COMPONENTS"];
    for(auto comp: components) {
      auto settings = py::cast<py::dict>(comp.second);
      if(components.size()==1) { //a single component
        initializeValue(settings);
      } else {
        initializeComponent(std::string(PYCV_COMPONENTPREFIX)
                            +"-"+py::cast<std::string>(comp.first),
                            settings);
      }
    }

  } else if(initDict.contains("Value")) {
    py::dict settingsDict=initDict["Value"];
    initializeValue(settingsDict);
  } else {
    warning("  WARNING: by defaults components periodicity is not set and component is added without derivatives - see manual\n");
    addValue();
  }

// ----------------------------------------

  auto pyParseAtomList=[this,&initDict](const char*key)->std::vector<AtomNumber> {
    std::vector<AtomNumber> myatoms;
    parseAtomList(key,myatoms);
    if (myatoms.size()>0 && initDict.contains(key))
      error(std::string("you specified the same keyword ").append(key)+ " both in python and in the settings file");
    if(initDict.contains(key)) {
      auto atomlist=PLMD::Tools::getWords(
        py::str(initDict[key]).cast<std::string>(),
        "\t\n ,");
      interpretAtomList( atomlist, myatoms );
    }
    return myatoms;
  };

  std::vector<AtomNumber> atoms=pyParseAtomList("ATOMS");
  std::vector<AtomNumber> groupA=pyParseAtomList("GROUPA");
  std::vector<AtomNumber> groupB=pyParseAtomList("GROUPB");

  if(atoms.size() !=0 && groupA.size()!=0)
    error("you can choose only between using the neigbourlist OR the atoms");

  if(atoms.size()==0&& groupA.size()==0 && groupB.size()==0)
    error("At least one atom is required");

  if (atoms.size() != 0 && groupA.size() != 0)
    error("you can choose only between using the neigbourlist OR the atoms");

  if (atoms.size() == 0 && groupA.size() == 0 && groupB.size() == 0)
    error("At least one atom is required");

  auto pyParseFlag=[this,&initDict](const char*key)->bool {
    bool toRet;
    parseFlag(key, toRet);
    if(initDict.contains(key)) {
      bool defaultRet;
      keywords.getLogicalDefault(key,defaultRet);
      if (toRet!=defaultRet) {
        error(std::string("you specified the same keyword ").append(key)+ " both in python and in the settings file");
      }
      toRet = initDict[key].cast<bool>();
    }
    return toRet;
  };

  bool nopbc = pyParseFlag("NOPBC");
  pbc = !nopbc;


  if (groupA.size() > 0) {
    // parse the NL things only in the NL case
    bool dopair = pyParseFlag("PAIR");
    // this is a WIP

    bool serial = false;
    bool doneigh = pyParseFlag("NLIST");
    double nl_cut = 0.0;
    int nl_st = 0;
    if (doneigh) {
      //parse("NL_CUTOFF", nl_cut);
      pyParse("NL_CUTOFF", initDict, nl_cut);
      if (nl_cut <= 0.0)
        error("NL_CUTOFF should be explicitly specified and positive");
      pyParse("NL_STRIDE",initDict, nl_st);
      if (nl_st <= 0)
        error("NL_STRIDE should be explicitly specified and positive");
    }
    // endof WIP
    if (groupB.size() > 0) {
      if (doneigh)
        nl = Tools::make_unique<NeighborList>(
               groupA, groupB, serial, dopair, pbc, getPbc(), comm, nl_cut, nl_st);
      else
        nl = Tools::make_unique<NeighborList>(groupA, groupB, serial, dopair,
                                              pbc, getPbc(), comm);
    } else {
      if (doneigh)
        nl = Tools::make_unique<NeighborList>(groupA, serial, pbc, getPbc(),
                                              comm, nl_cut, nl_st);
      else
        nl = Tools::make_unique<NeighborList>(groupA, serial, pbc, getPbc(),
                                              comm);
    }
    requestAtoms(nl->getFullAtomList());
  } else {
    requestAtoms(atoms);
  }

  if (getNumberOfComponents()>1) {
    log.printf("  it is expected to return dictionaries with %d components\n",
               getNumberOfComponents());
  }

  log << "  Bibliography " << plumed.cite(PYTHONCV_CITATION) << "\n";
} catch (const py::error_already_set &e) {
  plumed_merror(e.what());
  //vdbg(e.what());
}

void PythonCVInterface::initializeValue(pybind11::dict &settingsDict) {
  log << "  will have a single component";
  bool withDerivatives=false;
  if(settingsDict.contains("derivative")) {
    withDerivatives=settingsDict["derivative"].cast<bool>();
  }
  if(withDerivatives) {
    addValueWithDerivatives();
    log << " WITH derivatives\n";
  } else {
    addValue();
    log << " WITHOUT derivatives\n";
  }
  valueSettings(settingsDict,getPntrToValue());
}
void PythonCVInterface::initializeComponent(const std::string&name,py::dict &settingsDict) {
  bool withDerivatives=false;
  if(settingsDict.contains("derivative")) {
    withDerivatives=settingsDict["derivative"].cast<bool>();
  }

  if(withDerivatives) {
    addComponentWithDerivatives(name);
    log << " WITH derivatives\n";
  } else {
    addComponent(name);
    log << " WITHOUT derivatives\n";
  }
  valueSettings(settingsDict,getPntrToComponent(name));
}

void PythonCVInterface::valueSettings(py::dict &settings, Value* valPtr) {
  if(settings.contains("period")) {
    if (settings["period"].is_none()) {
      valPtr->setNotPeriodic();
    } else {
      py::tuple t = settings["period"];
      if(t.size()!=2) {
        plumed_merror("period must have exactly 2 components");
      }
      //the ballad py::str(t[0]).cast<std::string>() is to not care about the type of input of the user
      std::string min=py::str(t[0]).cast<std::string>();
      std::string max=py::str(t[1]).cast<std::string>();
      valPtr->setDomain(min, max);
    }
  }
}

void PythonCVInterface::prepare() {
  if (nl) {
    if (nl->getStride() > 0) {
      if (firsttime || (getStep() % nl->getStride() == 0)) {
        requestAtoms(nl->getFullAtomList());
        invalidateList = true;
        firsttime = false;
      } else {
        requestAtoms(nl->getReducedAtomList());
        invalidateList = false;
        if (getExchangeStep())
          error("Neighbor lists should be updated on exchange steps - choose a "
                "NL_STRIDE which divides the exchange stride!");
      }
      if (getExchangeStep())
        firsttime = true;
    }
  }
  if (hasPrepare) {
    py::dict prepareDict = pyPrepare(this);
    if (prepareDict.contains("setAtomRequest")) {
      //should I use "interpretAtomList"?
      py::tuple t = prepareDict["setAtomRequest"];
      std::vector<PLMD::AtomNumber> myatoms;
      for (const auto &i : t) {
        auto at = PLMD::AtomNumber::index(i.cast<unsigned>());
        myatoms.push_back(at);
      }
      for (const auto &i : myatoms) {
        std::cout << i.index() << " ";
      }
      std::cout << "\n";
      requestAtoms(myatoms);
    }
  }
  // NB: the NL kewywords will be counted as error when using ATOMS
  checkRead();
}

void PythonCVInterface::update() {
  if(hasUpdate) {
    py::dict updateDict=pyUpdate(this);
    //See what to do here
  }
}

// calculator
void PythonCVInterface::calculate() {
  try {
    if (nl) {
      if (nl->getStride() > 0 && invalidateList) {
        nl->update(getPositions());
      }
    }
    // Call the function
    py::object r = py_fcn(this);
    if(getNumberOfComponents()>1) {		// MULTIPLE NAMED COMPONENTS
      calculateMultiComponent(r);
    } else { // SINGLE COMPONENT
      calculateSingleComponent(r);
    }

  } catch (const py::error_already_set &e) {
    plumed_merror(e.what());
  }
}

void PythonCVInterface::calculateSingleComponent(py::object &r) {
  readReturn(r, getPntrToValue());
}

void PythonCVInterface::readReturn(const py::object &r, Value* valPtr) {
  // Is there more than 1 return value?
  if (py::isinstance<py::tuple>(r)||py::isinstance<py::list>(r)) {
    // 1st return value: CV
    py::list rl=r.cast<py::list>();
    pycvComm_t value = rl[0].cast<pycvComm_t>();
    valPtr->set(value);
    auto natoms = getPositions().size();
    if (rl.size() > 1) {
      if(!valPtr->hasDerivatives())
        plumed_merror(valPtr->getName()+" was declared without derivatives, but python returned with derivatives");
      // 2nd return value: gradient: numpy array of (natoms, 3)
      py::array_t<pycvComm_t> grad(rl[1]);
      // Assert correct gradient shape
      if (grad.ndim() != 2 || grad.shape(0) != natoms || grad.shape(1) != 3) {
        log.printf("Error: wrong shape for the gradient return argument: should be "
                   "(natoms=%d,3), received %ld x %ld\n",
                   natoms, grad.shape(0), grad.shape(1));
        plumed_merror("Python CV returned wrong gradient shape error");
      }
      // To optimize, see "direct access"
      // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
      for (unsigned i = 0; i < natoms; i++) {
        Vector3d gi(grad.at(i, 0), grad.at(i, 1), grad.at(i, 2));
        setAtomsDerivatives(valPtr, i, gi);
      }
    } else if (valPtr->hasDerivatives())
      plumed_merror(valPtr->getName()+" was declared with derivatives, but python returned none");

    if (rl.size() > 2) {
      if(!valPtr->hasDerivatives())
        plumed_merror(valPtr->getName()+" was declared without derivatives, but python returned with box derivatives");
      py::array_t<pycvComm_t> pyBoxDev(rl[2]);
      // expecting the box derivatives
      Tensor boxDev;
      if (pyBoxDev.ndim() == 2 &&
          (pyBoxDev.shape(0) == 3 && pyBoxDev.shape(1) == 3)) { // boxDev is 3x3
        boxDev =
          Tensor({pyBoxDev.at(0, 0), pyBoxDev.at(0, 1), pyBoxDev.at(0, 2),
                  pyBoxDev.at(1, 0), pyBoxDev.at(1, 1), pyBoxDev.at(1, 2),
                  pyBoxDev.at(2, 0), pyBoxDev.at(2, 1), pyBoxDev.at(2, 2)});
      } else if (pyBoxDev.ndim() == 1 && pyBoxDev.shape(0) == 9) {
        boxDev = Tensor({pyBoxDev.at(0), pyBoxDev.at(1), pyBoxDev.at(2),
                         pyBoxDev.at(3), pyBoxDev.at(4), pyBoxDev.at(5),
                         pyBoxDev.at(6), pyBoxDev.at(7), pyBoxDev.at(8)});
      } else {
        log.printf(
          "Error: wrong shape for the box derivatives return argument: "
          "should be (size 3,3 or 9), received %ld x %ld\n",
          natoms, pyBoxDev.shape(0), pyBoxDev.shape(1));
        error("Python CV returned wrong box derivatives shape error");
      }
      setBoxDerivatives(valPtr, boxDev);
    } else if (valPtr->hasDerivatives())
      warning(valPtr->getName()+" was declared with derivatives, but python returned no box derivatives");
  } else {
    // Only value returned. Might be an error as well.
    if (valPtr->hasDerivatives())
      warning(BIASING_DISABLED);
    pycvComm_t value = r.cast<pycvComm_t>();
    valPtr->set(value);
  }
  //TODO: is this ok?
  if (!pbc)
    setBoxDerivativesNoPbc(valPtr);
}


void PythonCVInterface::calculateMultiComponent(py::object &r) {

  const auto nc = getNumberOfComponents();
  if (py::isinstance<py::dict>(r)) {
    py::dict dataDict = r.cast<py::dict>(); // values
    for(int i=0; i < nc; ++i) {
      auto component=getPntrToComponent(i);
      //get the without "label.prefix-"
      std::string key=component->getName().substr(
                        2 + getLabel().size()
                        +PYCV_COMPONENTPREFIX.size());
      if (dataDict.contains(key.c_str()))
        readReturn(dataDict[key.c_str()], component);
      else
        plumed_merror( "python did not returned " << key );
    }
  } else {
    // In principle one could handle a "list" return case.
    error("Sorry, multi-components needs to return dictionaries");
  }
}



NeighborList &PythonCVInterface::getNL() { return *nl; }
} // namespace pycvs
} // namespace PLMD
