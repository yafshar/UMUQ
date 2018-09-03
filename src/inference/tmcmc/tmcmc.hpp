#ifndef UMUQ_TMCMC_H
#define UMUQ_TMCMC_H

#include "data/stdata.hpp"
#include "data/database.hpp"
#include "data/runinfo.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "io/io.hpp"

template <typename T, class F = fitFunction<T>>
class tmcmc
{
public:
  tmcmc();

  bool init();

  inline bool setInputFileName(char const *fileName = "tmcmc.par")
  {
    inputFilename = std::string(fileName);
    io f;
    return f.isFileExist(inputFilename);
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

  //! fit function object
  F fit;

private:
  //! stream data for getting the problem size and variables from the input file
  stdata<T> Data;

  // psrandom<> rand;
};

template <typename T, class F>
tmcmc<T, F>::tmcmc() : inputFilename("tmcmc.par")
{
}

template <typename T, class F>
bool tmcmc<T, F>::init()
{
  // Read the input problem size and variables from an input file
  if (Data.load(inputFilename))
  {
    // Creating a database based on the read information
    currentData = std::move(database<T>(Data.nDim, Data.maxGenerations));

    // Creating a database based on the read information
    fullData = std::move(database<T>(Data.nDim, Data.maxGenerations));

    //! Creating the run inofrmation data
    runData = std::move(runinfo<T>(Data.nDim, Data.maxGenerations));

    if (!fit.init())
    {
      return false;
    }

    return true;
  }
  return false;
}

#endif
