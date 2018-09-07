#ifndef UMUQ_TMCMC_H
#define UMUQ_TMCMC_H

#include "data/datatype.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "inference/prior/priordistribution.hpp"
#include "io/io.hpp"

namespace umuq
{

/*! \namespace tmcmc
 * \brief Namespace containing all the functions for TMCMC algorithm
 *
 */
namespace tmcmc
{

template <typename T>
std::unique_ptr<torcEnvironment<T>> torc;

/*! \class tmcmc
 * \brief 
 * 
 * \tparam T  T Data type
 * \tparam F  Function type, which is used in fit function (default FITFUN_T<T>) 
 */
template <typename T, class F = FITFUN_T<T>>
class tmcmc
{
public:
  tmcmc();

  ~tmcmc();

  /*!
   * \brief Set the Input File Name
   * 
   * \param fileName Input file name 
   * 
   * \return true 
   * \return false If the file does not exist
   */
  inline bool setInputFileName(char const *fileName = "");

  /*!
   * \brief Reset the managed object to the correct size, which is read from input file name
   * 
   * \param fileName Input file name 
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  bool reset(char const *fileName = "");

  /*!
   * \brief Set the Fitting Function object to be used
   * 
   * NOTE:
   * std::move is used so that the Fit Function object fitFun may be "moved from", 
   * i.e. allowing the efficient transfer of resources from fitFun to tmcmc member object. 
   * Thus, the fitFun object is not accessible after calling this function!
   * 
   * \param fitFun Fitting Function object 
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(fitFunction<T, F> &fitFun);

  /*!
   * \brief Set the fitting Function to be used
   * 
   * \param Fun  Fitting Function of type (class F)
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(F &Fun);
  inline bool setFitFunction(F const &Fun);

  /*!
   * \brief Setting both the Init Function & fitting Function members of Fit Function member
   * 
   * \param InitFun  Initilization function which has the fixed bool type
   * \param Fun      Fitting Function of type (class F)
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(std::function<bool()> &InitFun, F &Fun);
  inline bool setFitFunction(std::function<bool()> const &InitFun, F const &Fun);

  /*!
   * \brief Initilize the algorithm and set up the TORC environemnt
   * 
   * \param fileName Input file name 
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  bool init(char const *fileName = "");

  bool iterate();

private:
  inline bool iterate0();

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
  fitFunction<T, F> fitfun;

private:
  //! stream data for getting the problem size and variables from the input file
  stdata<T> Data;

  //! Prior distrdibution object
  priorDistribution<T> prior;

  //! pseudo-random numbers
  psrandom<T> prng;
};

template <typename T, class F>
tmcmc<T, F>::tmcmc() : inputFilename("input.par")
{
  torc<T>.reset(nullptr);
}

template <typename T, class F>
tmcmc<T, F>::~tmcmc()
{
  if (torc<T>)
  {
    torc<T>->TearDown();
  }
}

template <typename T, class F>
inline bool tmcmc<T, F>::setInputFileName(char const *fileName)
{
  //! Create an instance of the io object
  umuq::io f;
  if (strlen(fileName) > 0)
  {
    inputFilename = std::string(fileName);
    return f.isFileExist(fileName);
  }
  return f.isFileExist(inputFilename);
}

template <typename T, class F>
bool tmcmc<T, F>::reset(char const *fileName)
{
  if (setInputFileName(fileName))
  {
    // Read the input problem size and variables from an input file
    if (Data.load(inputFilename))
    {
      // Creating a database based on the read information
      currentData = std::move(database<T>(Data.nDim, Data.populationSize));

      { //! Compute the total size of data points
        auto sum = std::accumulate(Data.eachPopulationSize.begin(), Data.eachPopulationSize.end(), 0);

        // Creating a database based on the read information
        fullData = std::move(database<T>(Data.nDim, sum));
      }

      //! Creating the running inofrmation data
      runData = std::move(runinfo<T>(Data.nDim, Data.maxGenerations));

      //! Seed the PRNG
      if (!prng.setSeed(Data.seed))
      {
        UMUQWARNING("The Psudo random number generator has been seeded & initilized before!")
      }

      //! Construct a prior Distribution object
      prior = std::move(priorDistribution<T>(Data.nDim, Data.priorType));

      //! Set the prior parameters
      return prior.set(Data.priorParam1, Data.priorParam2, Data.compositePriorDistribution);
    }
    UMUQFAILRETURN("Failed to initilize the data from Input file!");
  }
  UMUQFAILRETURN("Requested File does not exist in the current PATH!!");
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(fitFunction<T, F> &fitFun)
{
  if (fitFun)
  {
    fitfun = std::move(fitFun);
    return true;
  }
  UMUQFAILRETURN("Function is not assigned!");
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(F &Fun)
{
  return fitfun.setFitFunction(Fun);
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(F const &Fun)
{
  return fitfun.setFitFunction(Fun);
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(std::function<bool()> &InitFun, F &Fun)
{
  return fitfun.set(InitFun, Fun);
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(std::function<bool()> const &InitFun, F const &Fun)
{
  return fitfun.set(InitFun, Fun);
}

template <typename T, class F>
bool tmcmc<T, F>::init(char const *fileName)
{
  if (fitfun)
  {
    if (setInputFileName(fileName))
    {
      //! Initializa the fitting function
      fitfun.init();

      //! Create a torc environment object
      torc<T>.reset(new torcEnvironment<T>);

      //! Set up the TORC environemnt
      torc<T>->SetUp();

      //! Set the State of psudo random number generating object
      return prng.setState();
    }
    UMUQFAILRETURN("Input file does not exist!");
  }
  UMUQFAILRETURN("Fitting function is not assigned! \n Fitting function must be set before calling this routine!");
}

template <typename T, class F>
inline bool tmcmc<T, F>::iterate0()
{
  //! Check if the restart file is available or not
  return runData.load() ? currentData.load("", runData.Generation) : false;
}

template <typename T, class F>
bool tmcmc<T, F>::iterate()
{
  //! If it is a fresh run, or the restart data is corrupted or is not available
  if (!iterate0())
  {
    for (int i = 0; i < Data.eachPopulationSize[0]; i++)
    {
    }
  }
}

} // namespace tmcmc
} // namespace umuq

#endif
