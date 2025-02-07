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

template <class T>
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

  ParallelTaskManager(ActionWithVector* av);
/// Setup an array to hold all the indices that are used for derivatives
  void setupIndexList( const std::vector<std::size_t>& ind );
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

template <class T>
ParallelTaskManager<T>::ParallelTaskManager(ActionWithVector* av):
  action(av),
  comm(av->comm),
  ismatrix(false),
  myinput() {
  ActionWithMatrix* am=dynamic_cast<ActionWithMatrix*>(av);
  if(am) {
    ismatrix=true;
  }
}

template <class T>
void ParallelTaskManager<T>::setMode( const unsigned val ) {
  myinput.mode = val;
}

template <class T>
void ParallelTaskManager<T>::setPbcFlag( const bool val ) {
  myinput.usepbc = val;
}

template <class T>
void ParallelTaskManager<T>::setupIndexList( const std::vector<std::size_t>& ind ) {
  // myinput.indices.resize( ind.size() ); for(unsigned i=0; i<ind.size(); ++i) myinput.indices[i] = ind[i];
}

template <class T>
void ParallelTaskManager<T>::runAllTasks( const unsigned& natoms ) {

  // Clear matrix bookeeping arrays
  // if( ismatrix && stride>1 ) clearMatrixBookeeping();

  // Get the list of active tasks
  std::vector<unsigned> & partialTaskList( action->getListOfActiveTasks( action ) );
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
  myinput.task_index=3;
  std::cerr <<myinput.task_index <<std::endl;
  unsigned t =  myinput.task_index;
#pragma acc data copy(t)// copyin(nactive_tasks )
  {


    printf("t: %d\n",t);

    // #pragma acc parallel loop gang //private(myinput)
#pragma  acc parallel loop  reduction(+:t)
    for(unsigned i=0; i<10000; i++) {
      // mode +=i;
      // auto myval = MultiValue(ncomponents, nderivatives, natoms);
      // myinput.task_index = partialTaskList[i];
      t+=i;
      // runTask(partialTaskList[i], mode, myval );

      // Transfer the data to the values
      // if( !ismatrix ) transferToValue( partialTaskList[i], myval );

      // Clear the value
      // myval.clearAll();
    }
  }
  {
    myinput.task_index=t;
  }
  printf("%d\n",myinput.task_index);

  std::cerr <<myinput.task_index <<std::endl;
}

template <class T>
void runTask( const ParallelActionsInput& locinp, MultiValue& myvals ) {

// void ParallelTaskManager<T>::runTask( unsigned const task_index, unsigned const mode, MultiValue& myvals ) {
  myvals.setTaskIndex(locinp.task_index);
  // T::performTask( locinp, myvals );
}

template <class T>
void ParallelTaskManager<T>::transferToValue( unsigned task_index, const MultiValue& myvals ) const {
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
