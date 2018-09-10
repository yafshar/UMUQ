#ifndef UMUQ_TMCMC_H
#define UMUQ_TMCMC_H

#include "data/datatype.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "inference/prior/priordistribution.hpp"
#include "io/io.hpp"
#include "misc/funcallcounter.hpp"

namespace umuq
{

/*! \namespace tmcmc
 * \brief Namespace containing all the functions for TMCMC algorithm
 *
 */
namespace tmcmc
{

/*!
 * \brief TORC environemnt object
 * 
 * \tparam T Data type
 */
template <typename T>
std::unique_ptr<torcEnvironment<T>> torc;

//! Create an instance of funcallcounter object
funcallcounter fc;

template <typename T, class F>
void tmcmcInitTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints, T *Fvalue, int const *nFvalue, int const *WorkInformation);

template <typename T, class F>
void tmcmcUpdateTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints, T *Fvalue, int const *nFvalue, int const *WorkInformation);

template <typename T, class F>
using TMCMTASKTYPE = void (*)(long long const, T const *, int const *, T *, int const *, int const *);

//! True if TMCMC Tasks have been registered, and false otherwise (logical).
template <typename T, class F>
static bool tmcmcTaskRegistered = false;

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

  bool prepareNewGeneration(database<T> &leaders);

private:
  inline bool iterate0();

public:
  //! Input file name
  std::string inputFilename;

  //! stream data for getting the problem size and variables from the input file
  stdata<T> Data;

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
  //! Prior distrdibution object
  priorDistribution<T> prior;

  //! pseudo-random numbers
  psrandom<T> prng;

  //! Sample points
  std::vector<T> samplePoints;

  //! Function values
  std::vector<T> Fvalue;

  //! Array of the work inofmrtaion
  int workInformation[3];

  //! Muex object
  std::mutex m;
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
  //! Check to see if the input file exists in the PATH or not?
  if (setInputFileName(fileName))
  {
    //! Read the input problem size and variables from an input file
    if (Data.load(inputFilename))
    {
      //! Creating a database based on the read information
      currentData = std::move(database<T>(Data.nDim, Data.populationSize));

      //! Creating the running inofrmation data
      runData = std::move(runinfo<T>(Data.nDim, Data.maxGenerations));

      //! Seed the PRNG
      if (!prng.setSeed(Data.seed))
      {
        UMUQWARNING("The Pseudo random number generator has been seeded & initilized before!")
      }

      //! Assign the correct size
      samplePoints.resize(Data.nDim);

      Fvalue.resize(1);

      //! Construct a prior Distribution object
      prior = std::move(priorDistribution<T>(Data.nDim, Data.priorType));

      //! Set the prior parameters
      return prior.set(Data.priorParam1, Data.priorParam2, Data.compositePriorDistribution);
    }
    UMUQFAILRETURN("Failed to initilize the data from the Input file!");
  }
  UMUQFAILRETURN("Input file for the input TMCMC parameter does not exist in the current PATH!!");
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
      //! Initializa the fitting function, this should be done before initializaing the TORC environment
      fitfun.init();

      //! Create a torc environment object
      torc<T>.reset(new torcEnvironment<T>);

      //! Register tasks
      {
        std::lock_guard<std::mutex> lock(m);

        // Check if TMCMC tasks have been not been registered, do it
        if (!tmcmcTaskRegistered<T, F>)
        {
          torc_register_task((void *)tmcmcInitTask<T, F>);
          torc_register_task((void *)tmcmcUpdateTask<T, F>);
          tmcmcTaskRegistered<T, F> = true;
        }
      }

      //! Set up the TORC environemnt
      torc<T>->SetUp();

      //! Set the State of pseudo random number generating object
      if (prng.setState())
      {
        //! Set the Random Number Generator object in prior
        return prior.setRandomGenerator(&prng);
      }
      UMUQFAILRETURN("Failed to initialize the PRNG or set the state of that!");
    }
    UMUQFAILRETURN("Input file for the input TMCMC parameter does not exist in the current PATH!!");
  }
  UMUQFAILRETURN("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
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
    //! Generation number
    workInformation[0] = runData.Generation;

    //! Step number
    workInformation[2] = 0;

    int nFvalue = Fvalue.size();

    for (int i = 0; i < Data.eachPopulationSize[0]; i++)
    {
      //! Sample number
      workInformation[1] = i;

      //! Create the input sample points from the prior distribution
      prior.sample(samplePoints);

      //! Create tasks
      torc_create(-1, (void (*)())tmcmcInitTask<T, F>, 6,
                  1, MPIDatatype<long long>, CALL_BY_REF,
                  Data.nDim, MPIDatatype<T>, CALL_BY_COP,
                  1, MPIDatatype<int>, CALL_BY_COP,
                  1, MPIDatatype<T>, CALL_BY_COP,
                  1, MPIDatatype<int>, CALL_BY_COP,
                  3, MPIDatatype<int>, CALL_BY_COP,
                  reinterpret_cast<long long>(this), samplePoints.data(),
                  &Data.nDim, Fvalue.data(), &nFvalue, workInformation);
    }

    torc_enable_stealing();
    torc_waitall();
    torc_disable_stealing();

    fc.count();

    std::cout << "server: Generation " << runData.Generation << ": total elapsed time = "
              << "secs, generation elapsed time = "
              << "secs for function calls = " << fc.getLocalFunctionCallsNumber() << std::endl;

    //! Reset the local counter to zero
    fc.reset();

    if (Data.saveData)
    {
      if (!runData.save("curgen_db", runData.Generation))
      {
        UMUQFAILRETURN("Failed to write down the current data information!");
      }
    }

    runData.save();
  }

  //! Create memory for leader selection
  database<T> leaders(Data.nDim, Data.eachPopulationSize[runData.Generation]);
}

/* process curgen_db -> calculate statitics */
/* compute probs based on F values */
/* draw new samples (nchains or user-specified) */
/* find unique samples: fill the (new) leaders table */
/* count how many times they appear -> nsteps */
/* return the new sample size (number of chains) */
template <typename T, class F>
bool tmcmc<T, F>::prepareNewGeneration(database<T> &leaders)
{
  //! Create an instance of the statistics object
  stats s;

  //! Compute vectors of mean and standard deviation for each dimension
  std::vector<T> mean(currentData.ndimParray);
  std::vector<T> stddev(currentData.ndimParray);

  int nSize = static_cast<int>(currentData.getSize()) * currentData.ndimParray;

  for (int i = 0; i < currentData.ndimParray; i++)
  {
    mean[i] = s.mean<T, T>(currentData.Parray.get() + i, nSize, currentData.ndimParray);
    stddev[i] = s.stddev<T, T>(currentData.Parray.get() + i, nSize, currentData.ndimParray, mean[i]);
  }

#ifdef DEBUG
  io f;
  std::cout << "Generation No = " << runData.Generation << std::endl;
  f.printMatrix<T>("Means", mean.data(), 1, currentData.ndimParray);
  f.printMatrix<T>("Stddev", stddev.data(), 1, currentData.ndimParray);
#endif

  //! Now we check to find unique points
  

}

template <typename T, class F>
void tmcmcInitTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints, T *Fvalue, int const *nFvalue, int const *WorkInformation)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<T, F> *>(TMCMCObj);

  //! If we have set the fittiting function
  if (tmcmcObj->fitfun)
  {
    int const nSamples = *nSamplePoints;
    int const nFunvals = *nFvalue;

    //! Create an array of work information
    std::vector<int> workInformation{WorkInformation, WorkInformation + 3};

    //! Create an array of sample points
    std::vector<T> samplePoints{SamplePoints, SamplePoints + nSamples};

    //! Increment function call counter
    fc.increment();

    //! Call the fitting function
    Fvalue[0] = tmcmcObj->fitfun.f(samplePoints.data(), nSamples, Fvalue, nFunvals, workInformation.data());

    // T *fv = Fvalue;
    // fv += (nFunvals > 1 ? 1 : 0);

    //! Update the data
    tmcmcObj->currentData.update(samplePoints.data(), Fvalue[0]);

    return;
  }
  UMUQFAIL("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

template <typename T, class F>
void tmcmcUpdateTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints, T *Fvalue, int const *nFvalue, int const *WorkInformation)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<T, F> *>(TMCMCObj);

  //! If we have set the fittiting function
  if (tmcmcObj->fitfun)
  {
    return;
  }
  UMUQFAIL("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

} // namespace tmcmc
} // namespace umuq

#endif
