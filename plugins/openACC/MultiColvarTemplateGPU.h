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
#ifndef __PLUMED_colvar_MultiColvarTemplateGPU_h
#define __PLUMED_colvar_MultiColvarTemplateGPU_h

#include "plumed/core/ActionWithVector.h"
namespace PLMD {
namespace colvar {
using atomBlocks = std::vector< std::vector<unsigned> > ;
} // namespace colvar
} // namespace PLMD
#include "ParallelTaskManager.h"
#include <string>
namespace PLMD {

// template < typename T, std::size_t N=dynamic_extent>
template < typename T, std::size_t N>
class View {
  T *ptr_;
  // const std::size_t size_;
public:
  View(T* p) :ptr_(p) {}
  constexpr size_t size()const {
    return N;
  }
  static constexpr size_t extent = N;
  T & operator[](size_t i) {
    return ptr_[i];
  }
  const T & operator[](size_t i) const {
    return ptr_[i];
  }
};

class Colvar;

namespace colvar {

template <std::size_t N>
struct
  ColvarInput {
  unsigned mode;
  // const Pbc& pbc;
  const View<Vector,N> pos;
  const View<double,N> mass;
  const View<double,N> charges;
  ColvarInput(unsigned m, Vector* p, double* w,
              double* q ):
    mode(m),
    //pbc(box),
    pos(p),
    mass(w),
    charges(q) {
  }
};
template <std::size_t N>
struct ColvarOutput {
  View<double,N> vals;
  ColvarOutput(              double* v ):
    vals(v) {}
};

template <typename theAction>
class MultiColvarTemplateGPU : public ActionWithVector {
private:
/// The parallel task manager
  ParallelTaskManager<theAction> taskmanager;
/// An index that decides what we are calculating
  unsigned mode;
/// Are we using pbc to calculate the CVs
  bool usepbc;
/// Do we reassemble the molecule
  bool wholemolecules;
/// Blocks of atom numbers
//better use plmd::matrix<unsigned>, for memory compactness
  atomBlocks ablocks;
public:
  static void registerKeywords(Keywords&);
  explicit MultiColvarTemplateGPU(const ActionOptions&);
  unsigned getNumberOfDerivatives() override ;
  unsigned getNumberOfAtomsPerTask() const ;
  void addValueWithDerivatives( const std::vector<unsigned>& shape=std::vector<unsigned>() ) ;
  void addComponentWithDerivatives( const std::string& name, const std::vector<unsigned>& shape=std::vector<unsigned>() )  ;
  void getInputData(ParallelActionsMV& inputdata ) const  ;
  void performTask( const unsigned&, MultiValue& ) const  ;
  void calculate() override;
  static void performTask( unsigned mode,
                           unsigned task_index,
                           //const ParallelActionsInput& input,
                           std::vector<double> & mass,
                           std::vector<double> & charge,
                           std::vector<Vector> & fpositions
                           /*,MultiValue& myvals */);
};

template <typename theAction>
void MultiColvarTemplateGPU<theAction>::registerKeywords(Keywords& keys ) {
  theAction::registerKeywords( keys );
  keys.add("optional","MASK","the label for a sparse matrix that should be used to determine which elements of the matrix should be computed");
  unsigned nkeys = keys.size();
  for(unsigned i=0; i<nkeys; ++i) {
    if( keys.style( keys.getKeyword(i), "atoms" ) ) {
      keys.reset_style( keys.getKeyword(i), "numbered" );
    }
  }
  if( keys.outputComponentExists(".#!value") ) {
    keys.setValueDescription("vector","the " + keys.getDisplayName() + " for each set of specified atoms");
  }
}

template <typename theAction>
MultiColvarTemplateGPU<theAction>::MultiColvarTemplateGPU(const ActionOptions&ao):
  Action(ao),
  ActionWithVector(ao),
  taskmanager(this),
  mode(0),
  usepbc(true),
  wholemolecules(false) {
  std::vector<AtomNumber> all_atoms;
  if( getName()=="POSITION_VECTOR" || getName()=="MASS_VECTOR" || getName()=="CHARGE_VECTOR" ) {
    parseAtomList( "ATOMS", all_atoms );
  }
  if( all_atoms.size()>0 ) {
    ablocks.resize(1);
    ablocks[0].resize( all_atoms.size() );
    //std::iota(ablocks[0].begin(),ablocks[0].end(),0u);
    for(unsigned i=0; i<all_atoms.size(); ++i) {
      ablocks[0][i] = i;
    }
  } else {
    std::vector<AtomNumber> t;
    for(int i=1;; ++i ) {
      theAction::parseAtomList( i, t, this );
      if( t.empty() ) {
        break;
      }

      if( i==1 ) {
        ablocks.resize(t.size());
      }
      if( t.size()!=ablocks.size() ) {
        std::string ss;
        Tools::convert(i,ss);
        error("ATOMS" + ss + " keyword has the wrong number of atoms");
      }
      for(unsigned j=0; j<ablocks.size(); ++j) {
        ablocks[j].push_back( ablocks.size()*(i-1)+j );
        all_atoms.push_back( t[j] );
      }
      t.resize(0);
    }
  }
  std::cerr << "ablocks: " <<ablocks.size()<< "\n";
  for (auto i=0u; i<ablocks.size() ; ++i) {
    std::cerr << "ablocks["<<i<<"]: " <<ablocks[i].size()<< "\n";
  }
  if( all_atoms.size()==0 ) {
    error("No atoms have been specified");
  }
  requestAtoms(all_atoms);
  if( keywords.exists("NOPBC") ) {
    bool nopbc=!usepbc;
    parseFlag("NOPBC",nopbc);
    usepbc=!nopbc;
  }
  if( keywords.exists("WHOLEMOLECULES") ) {
    parseFlag("WHOLEMOLECULES",wholemolecules);
    if( wholemolecules ) {
      usepbc=false;
    }
  }
  if( usepbc ) {
    log.printf("  using periodic boundary conditions\n");
  } else {
    log.printf("  without periodic boundary conditions\n");
  }

  // Setup the values
  mode = theAction::getModeAndSetupValues( this );
  // This sets up an array in the parallel task manager to hold all the indices
  std::vector<std::size_t> ind( ablocks.size()*ablocks[0].size() );
  for(unsigned i=0; i<ablocks[0].size(); ++i) {
    for(unsigned j=0; j<ablocks.size(); ++j) {
      ind[i*ablocks.size() + j] = ablocks[j][i];
    }
  }
  // Sets up the index list in the task manager
  taskmanager.setupIndexList( ind );
  taskmanager.setPbcFlag( usepbc );
  taskmanager.setMode( mode );
}

template <typename theAction>
unsigned MultiColvarTemplateGPU<theAction>::getNumberOfDerivatives() {
  return 3*getNumberOfAtoms()+9;
}

template <typename theAction>
void MultiColvarTemplateGPU<theAction>::calculate() {
  if( wholemolecules ) {
    makeWhole();
  }
  // setForwardPass(true);
  //sorry, but this is needed to make it run on master
  taskmanager.setInputData(ablocks, getPositions(),getMasses(),getCharges());
  taskmanager.runAllTasks( ablocks.size() );
  // setForwardPass(false);
}

template <typename theAction>
void MultiColvarTemplateGPU<theAction>::addValueWithDerivatives( const std::vector<unsigned>& shape ) {
  std::vector<unsigned> s(1);
  s[0]=ablocks[0].size();
  addValue( s );
}

template <typename theAction>
void MultiColvarTemplateGPU<theAction>::addComponentWithDerivatives( const std::string& name, const std::vector<unsigned>& shape ) {
  std::vector<unsigned> s(1);
  s[0]=ablocks[0].size();
  addComponent( name, s );
}

template <typename theAction>
unsigned MultiColvarTemplateGPU<theAction>::getNumberOfAtomsPerTask() const {
  return ablocks.size();
}


template <typename theAction>
void MultiColvarTemplateGPU<theAction>::performTask( const unsigned& task_index, MultiValue& myvals ) const {
  // Retrieve the positions
  std::vector<double> & mass( myvals.getTemporyVector(0) );
  std::vector<double> & charge( myvals.getTemporyVector(1) );
  std::vector<Vector> & fpositions( myvals.getFirstAtomVector() );
  for(unsigned i=0; i<ablocks.size(); ++i) {
    fpositions[i] = getPosition( ablocks[i][task_index] );
    mass[i]=getMass( ablocks[i][task_index] );
    charge[i]=getCharge( ablocks[i][task_index] );
  }
  std::vector<std::size_t> der_indices( ablocks.size() );
  for(unsigned i=0; i<der_indices.size(); ++i) {
    der_indices[i] = ablocks[i][task_index];
  }
  // performTask( mode, der_indices, doNotCalculateDerivatives(), usepbc, myvals );
}

template <typename theAction>
void MultiColvarTemplateGPU<theAction>::getInputData( ParallelActionsMV& inputdata ) const {
  unsigned ntasks = ablocks[0].size();
  if( inputdata.mass.size()!=ablocks.size()*ntasks ) {
    inputdata.fpositions.resize(ablocks.size()*ntasks);
    inputdata.mass.resize(ablocks.size()*ntasks);
    inputdata.mass.resize(ablocks.size()*ntasks);
  }
  std::size_t k=0;
  for(unsigned i=0; i<ntasks; ++i) {
    for(unsigned j=0; j<ablocks.size(); ++j) {
      Vector mypos( getPosition( ablocks[j][i] ) );
      inputdata.fpositions[k]=mypos;
      inputdata.mass[k] = getMass( ablocks[j][i] );
      inputdata.mass[k] = getCharge( ablocks[j][i] );
      ++k;
    }
  }
}

}
}
#endif
