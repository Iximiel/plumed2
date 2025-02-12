/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2017-2023 The plumed team
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
#ifndef __PLUMED_core_ParallelTaskManager_h
#define __PLUMED_core_ParallelTaskManager_h

#include <iostream>
#include "plumed/tools/Communicator.h"
#include "plumed/core/ActionWithVector.h"
#include "plumed/core/ActionWithMatrix.h"
#include "plumed/tools/OpenMP.h"
#define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " << (__VA_ARGS__) << '\n'
namespace PLMD {

struct ParallelActionsInput {
  bool usepbc;
  bool noderiv;

  unsigned mode;
  unsigned task_index;
/// This holds indices for creating derivatives
  // std::vector<std::size_t> indices;
/// This holds all the input data that is required to calculate all values for all tasks
  // std::vector<double> inputdata;
  ParallelActionsInput(  ) : usepbc(false), noderiv(false), mode(0), task_index(0) {}
};

struct ParallelActionsMV {
  std::vector<Vector> fpositions;
  std::vector<double> mass;
  std::vector<double> charge;
};

template <typename theAction>
class ParallelTaskManager {
private:
/// The underlying action for which we are managing parallel tasks
  ActionWithVector* action;
/// The MPI communicator
  Communicator& comm;
/// Is this an action with matrix
  bool ismatrix;
/// The buffer that we use (we keep a copy here to avoid resizing)
  std::vector<double> buffer;
/// A tempory vector of MultiValue so we can avoid doing lots of resizes
  std::vector<MultiValue> myvals;
public:
/// An action to hold data that we pass to and from the static function
  ParallelActionsInput myinput;
  ParallelActionsMV mydata;
  ParallelTaskManager(ActionWithVector* av);
/// Setup an array to hold all the indices that are used for derivatives
  void setupIndexList( const std::vector<std::size_t>& ind );

  void setInputData(const colvar::atomBlocks & ablocks,
                    const std::vector <Vector>& positions,
                    const std::vector <double>& masses,
                    const std::vector <double>& charges ) {
    const unsigned ntasks = ablocks[0].size();
    if( mydata.fpositions.size()!=ablocks.size()*ntasks ) {
      mydata.fpositions.resize(ablocks.size()*ntasks);
    }
    if( mydata.mass.size()!=ablocks.size()*ntasks ) {
      mydata.mass.resize(ablocks.size()*ntasks);
    }
    if( mydata.charge.size()!=ablocks.size()*ntasks ) {
      mydata.charge.resize(ablocks.size()*ntasks);
    }
    std::size_t k=0;
    for(unsigned i=0; i<ntasks; ++i) {
      for(unsigned j=0; j<ablocks.size(); ++j) {
        mydata.fpositions[k] = positions[ablocks[j][i]];
        mydata.mass[k] = masses [ablocks[j][i]] ;
        mydata.charge[k] = charges [ablocks[j][i]] ;
        ++k;
      }
    }
  }

/// Set the mode for the calculation
  void setMode( const unsigned val );
/// Set the value of the pbc flag
  void setPbcFlag( const bool val );
/// This runs all the tasks
  void runAllTasks( const unsigned& natoms=0 );
/// This runs each of the tasks
  static void runTask( const ParallelActionsInput& locinp, MultiValue& myvals );
/// Transfer the data to the Value
  void transferToValue(  unsigned task_index, const MultiValue& myvals ) const ;
};

template <typename theAction>
ParallelTaskManager <theAction>::ParallelTaskManager(ActionWithVector* av):
  action(av),
  comm(av->comm),
  ismatrix(false),
  myinput() {
  ActionWithMatrix* am=dynamic_cast<ActionWithMatrix*>(av);
  if(am) {
    ismatrix=true;
  }
}

template <typename theAction>
void ParallelTaskManager <theAction>::setMode( const unsigned val ) {
  myinput.mode = val;
}

template <typename theAction>
void ParallelTaskManager <theAction>::setPbcFlag( const bool val ) {
  myinput.usepbc = val;
}

template <typename theAction>
void ParallelTaskManager <theAction>::setupIndexList( const std::vector<std::size_t>& ind ) {
  // myinput.indices.resize( ind.size() ); for(unsigned i=0; i<ind.size(); ++i) myinput.indices[i] = ind[i];
}

template <typename theAction>
void ParallelTaskManager <theAction>::runAllTasks( const unsigned& natoms ) {

  // Clear matrix bookeeping arrays
  // if( ismatrix && stride>1 ) clearMatrixBookeeping();

  // Get the list of active tasks
  std::vector<unsigned>  partialTaskList( action->getListOfActiveTasks( action ) );
  unsigned nactive_tasks=partialTaskList.size();

  // Get the total number of streamed quantities that we need
  // Get size for buffer
  unsigned bufsize=0, nderivatives = 0;
  if( buffer.size()!=bufsize ) {
    buffer.resize( bufsize );
  }
  // Clear buffer
  buffer.assign( buffer.size(), 0.0 );

  // Get all the input data so we can broadcast it to the GPU
  myinput.noderiv = true;
  // action->getInputData( myinput.inputdata ); ////////////////////////////////////////////////////////////////////

  // Check if this is an actionWithMatrix object
  // const ActionWithMatrix* am = dynamic_cast<const ActionWithMatrix*>(action);
  // bool ismatrix=false; if(am) ismatrix=true;
  auto ncomponents = action->getNumberOfComponents();
  // auto mode = myinput.mode;
  myinput.task_index=0;
  std::cerr <<myinput.task_index <<std::endl;

  std::vector<double>  mass = mydata.mass;
  std::vector<double>  charge = mydata.charge;
  std::vector<Vector>  fpositions = mydata.fpositions;

  for (auto i=0; i< fpositions.size(); ++i) {
    std::cerr << i << " " << fpositions[i] << "\n";
  }
  auto fulltasksize=mass.size();
  vdbg(fulltasksize);
  vdbg(nactive_tasks);
  auto mode = myinput.mode;
  std::vector<double> values(3*nactive_tasks,0.0);
  auto atomsPerTask = 2;
#pragma acc data \
  copyin(nactive_tasks, \
         partialTaskList[0:nactive_tasks] ,\
         mass[0:fulltasksize], \
         charge[0:fulltasksize], \
         fpositions[0:fulltasksize],\
         mode, atomsPerTask) \
         copy(values[0:3*nactive_tasks])
  {


    // #pragma acc parallel loop gang //
#pragma  acc parallel loop
    for(unsigned i=0; i<nactive_tasks; i++) {
      auto task_index = partialTaskList[i];
      const std::size_t base = task_index * atomsPerTask;
      // mode +=i;
      // auto myval = MultiValue(ncomponents, nderivatives, natoms);


      // runTask(partialTaskList[i], mode, myval );

      theAction::calculateCV( {mode, fpositions.data()+base, mass.data()+base, charge.data()+base },
      {values.data()+base}
      //, derivs, virial
                            );
      // Transfer the data to the values
      // if( !ismatrix ) transferToValue( partialTaskList[i], myval );

      // Clear the value
      // myval.clearAll();
    }
  }
  std::cerr << "Values\n";
  for(unsigned i=0; i<nactive_tasks; i++) {
    std::cerr << i << " " << values[3*i+0] << " " << values[3*i+1] <<" " << values[3*i+2] << "\n";
  }

}

template <typename theAction>
void runTask( const ParallelActionsInput& locinp, MultiValue& myvals ) {

// void ParallelTaskManager <theAction>::runTask( unsigned const task_index, unsigned const mode, MultiValue& myvals ) {
  myvals.setTaskIndex(locinp.task_index);
  T::performTask( locinp, myvals );
}

template <typename theAction>
void ParallelTaskManager <theAction>::transferToValue( unsigned task_index, const MultiValue& myvals ) const {
  for(unsigned i=0; i<action->getNumberOfComponents(); ++i) {
    const Value* myval = action->getConstPntrToComponent(i);
    if( myval->hasDerivatives() || (action->getName()=="RMSD_VECTOR" && myval->getRank()==2) ) {
      continue;
    }
    Value* myv = const_cast<Value*>( myval );
    myv->set( task_index, myvals.get( i ) );
  }
}

}
#endif
