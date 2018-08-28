#ifndef UMUQ_TMCMC_H
#define UMUQ_TMCMC_H

#include "data/stdata.hpp"
#include "data/database.hpp"
#include "data/runinfo.hpp"

#include "io/io.hpp"

template <typename T>
class tmcmc
{
public:
  tmcmc();

  bool init();

  inline void setInputFileName(char const *fileName = "tmcmc.par")
  {
    inputFilename = std::string(fileName);
  }

public:
  //! Input file name
  std::string inputFilename;

  //! Current data
  database<T> currentData;

  //! Full data pointer
  database<T> fullData;

  //! Experimental data pointer
  database<T> expData;

  //! Running data
  runinfo<T> runData;

  //! stream data for getting the problem size and variables from the input file
  stdata<T> Data;
};

template <typename T>
tmcmc<T>::tmcmc() : inputFilename("tmcmc.par")
{
}

template <typename T>
bool tmcmc<T>::init()
{
  // Read the input problem size and variables from an input file
  if (Data.load(inputFilename))
  {

    // Creating a database based on the read information
    currentData = std::move(database<T>(Data.nDim, Data.maxGenerations));

    // Creating a database based on the read information
    fullData = std::move(database<T>(Data.nDim, Data.maxGenerations));

    if (runData.reset(Data.nDim, Data.maxGenerations))
    {
      return true;
    }

    return false;
  }
  return false;
}

#endif
