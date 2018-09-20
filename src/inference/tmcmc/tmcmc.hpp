#ifndef UMUQ_TMCMC_H
#define UMUQ_TMCMC_H

#include "data/datatype.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "numerics/multimin.hpp"
#include "inference/prior/priordistribution.hpp"
#include "io/io.hpp"
#include "misc/funcallcounter.hpp"
#include "tmcmcstats.hpp"

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
   * \brief Get the Input File Name object
   * 
   * \return char const* 
   */
  inline std::string getInputFileName();

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
   * \param InitFun  Initialization function which has the fixed bool type
   * \param Fun      Fitting Function of type (class F)
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(std::function<bool()> &InitFun, F &Fun);
  inline bool setFitFunction(std::function<bool()> const &InitFun, F const &Fun);

  /*!
   * \brief Initialize the algorithm and set up the TORC environemnt
   * 
   * \param fileName Input file name 
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  bool init(char const *fileName = "");

  /*!
   * \brief Check if the restart files are available or not. In case of being available, load, and update the data.
   * 
   * \returns true 
   * \returns false If this is a fresh start
   */
  inline bool restart();

  /*!
   * \brief Start of the TMCMC algorithm 
   * 
   * The original formulation of TMCMC constructs multiple intermediate target 
   * distributions as follows:
   * \f[
   *      f_j \propto p(D|\theta)^{\zeta_j}p(\theta) 
   * \f]
   * for \f$ j=1, \cdots,m\f$ and \f$ \zeta_1 <\zeta_2<\cdots<\zeta_m=1.\f$ 
   * Starting from samples drawn from the prior at stage \f$ j = 1. \f$ 
   *  
   * \returns true 
   * \returns false 
   */
  bool iterate0();

  /*!
   * \brief Start of the TMCMC algorithm 
   * 
   *  
   * \returns true 
   * \returns false 
   */
  bool iteratem();

  /*!
   * \brief Iterative part of the TMCMC algorithm 
   * 
   * The original formulation of TMCMC constructs multiple intermediate target 
   * distributions as follows:
   * \f[
   *      f_j \propto p(D|\theta)^{\zeta_j}p(\theta) 
   * \f]
   * for \f$ j=1, \cdots,m\f$ and \f$ \zeta_1 <\zeta_2<\cdots<\zeta_m=1.\f$ 
   * Starting from samples drawn from the prior at stage \f$ j = 1, \f$ the
   * posterior samples are obtained by updating the prior samples through 
   * \f$ m - 2 \f$ intermediate target distributions. 
   * 
   * When \f$ \zeta_j = 1, \f$ this procedure stops and the last set of samples
   * is approximately distributed according to the target posterior PDF, 
   * \f$ p(\theta | D) \f$. Through the m steps, the evidence is obtained. 
   *  
   * \returns true 
   * \returns false 
   */
  bool iterate();

  /*!
   * \brief Process current data, including, calculating the statistics 
   * and computing the probabilities based on function values, and draw 
   * new samples 
   * 
   * Preparing the new data generation includes:
   * - Process the current sampling points data
   * - Find unique sampling points
   * - Calculate the statistics of the data, including acceptance rate
   * - Compute probabilities based on function values for each sample point
   * - Select the new sample points leaders for propagating the chains and update the statistics
   * - Compute number of selections (steps) for each leader chain
   * - Update leaders information from current data information
   * 
   * \returns true 
   * \returns false 
   */
  bool prepareNewGeneration();

public:
  //! Input file name
  std::string inputFilename;

  //! Stream data for getting the problem size and variables from the input file
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
  //! Next generation data
  database<T> leadersData;

  //! TMCMC object for statistics
  tmcmcStats<T> tStats;

  //! Prior distribution object
  priorDistribution<T> prior;

  //! pseudo-random numbers
  psrandom<T> prng;

  //! Sample points
  std::vector<T> samplePoints;

  //! Function values
  std::vector<T> fValue;

  //! Array of the work information
  int workInformation[3];

  //! Local covariance with the size of [populationSize*nDim*nDim]
  std::vector<T> localCovariance;

  //  localCovariance(populationSize * nDim * nDim, T{})
  // for (int i = 0, l = 0; i < populationSize; i++)
  // {
  // 	for (int j = 0; j < nDim; j++)
  // 	{
  // 		for (int k = 0; k < nDim; k++, l++)
  // 		{
  // 			if (j == k)
  // 			{
  // 				localCovariance[l] = static_cast<T>(1);
  // 			}
  // 		}
  // 	}
  // }

  // localCovariance = std::move(other.localCovariance);
  // localCovariance = std::move(other.localCovariance);
  // localCovariance.swap(other.localCovariance);
  // localCovariance.clear();
  // localCovariance.resize(populationSize * nDim * nDim, T{});
  // for (int i = 0, l = 0; i < populationSize; i++)
  // {
  // 	for (int j = 0; j < nDim; j++)
  // 	{
  // 		for (int k = 0; k < nDim; k++, l++)
  // 		{
  // 			if (j == k)
  // 			{
  // 				localCovariance[l] = static_cast<T>(1);
  // 			}
  // 		}
  // 	}
  // }

  //! Mutex object
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
inline std::string tmcmc<T, F>::getInputFileName()
{
  return inputFilename;
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
      currentData = std::move(database<T>(Data.nDim, Data.lastPopulationSize));

      //! Creating the running information data
      runData = std::move(runinfo<T>(Data.nDim, Data.maxGenerations));

      //! Seed the PRNG
      if (!prng.setSeed(Data.seed))
      {
        UMUQWARNING("The Pseudo random number generator has been seeded & initialized before!")
      }

      //! Assign the correct size
      samplePoints.resize(Data.nDim);

      //! Function values
      fValue.resize(Data.lastPopulationSize);

      //! Initialize the tstats variable from read data
      tStats = std::move(tmcmcStats<T>(Data.options));

      //! Construct a prior Distribution object
      prior = std::move(priorDistribution<T>(Data.nDim, Data.priorType));

      //! Set the prior parameters
      return prior.set(Data.priorParam1, Data.priorParam2, Data.compositePriorDistribution);
    }
    UMUQFAILRETURN("Failed to initialize the data from the Input file!");
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
      //! Initialize the fitting function, this should be done before initializing the TORC environment
      fitfun.init();

      //! Create a torc environment object
      torc<T>.reset(new torcEnvironment<T>);

      //! Register tasks
      {
        std::lock_guard<std::mutex> lock(m);

        // Check if TMCMC tasks have not been registered, do it
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
        //! Set the Random Number Generator object in the prior
        return prior.setRandomGenerator(&prng);
      }
      UMUQFAILRETURN("Failed to initialize the PRNG or set the state of that!");
    }
    UMUQFAILRETURN("Input file for the input TMCMC parameter does not exist in the current PATH!!");
  }
  UMUQFAILRETURN("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

template <typename T, class F>
inline bool tmcmc<T, F>::restart()
{
  //! Check if the restart file is available or not
  return runData.load() ? currentData.load("", runData.currentGeneration) : false;
}

template <typename T, class F>
bool tmcmc<T, F>::iterate0()
{
  //! Check if it is a fresh run, or the restart data is corrupted or is not available
  if (!restart())
  {
    //! currentGeneration number
    workInformation[0] = runData.currentGeneration;

    //! Step number, this is the first step
    workInformation[2] = 0;

    //! Number of function value at each sampling point
    int nFvalue = 1;

    //! Loop through all the population size
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
                  1, MPIDatatype<T>, CALL_BY_RES,
                  1, MPIDatatype<int>, CALL_BY_COP,
                  3, MPIDatatype<int>, CALL_BY_COP,
                  reinterpret_cast<long long>(this), samplePoints.data(),
                  &Data.nDim, fValue.data() + i, &nFvalue, workInformation);
    }

    torc_enable_stealing();
    torc_waitall();
    torc_disable_stealing();

    //! Count the function calls
    fc.count();

    //! Print the summary
    std::cout << "server: currentGeneration " << runData.currentGeneration << ": total elapsed time = "
              << "secs, generation elapsed time = "
              << "secs for function calls = " << fc.getLocalFunctionCallsNumber() << std::endl;

    //! Reset the local function counter to zero
    fc.reset();

    //! Check if we should save the mid run data
    if (Data.saveData)
    {
      if (!currentData.save("curgen_db", runData.currentGeneration))
      {
        UMUQFAILRETURN("Failed to write down the current data information!");
      }
    }

    //! Running information for checkpoint restart
    return runData.save();
  }
  return true;
}

template <typename T, class F>
bool tmcmc<T, F>::iteratem()
{
  //! currentGeneration number
  workInformation[0] = runData.currentGeneration;

  //! Step number, this is the first step
  workInformation[2] = 0;

  //! Number of function value at each sampling point
  int nFvalue = 1;

  //! Loop through all the population size
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
                1, MPIDatatype<T>, CALL_BY_RES,
                1, MPIDatatype<int>, CALL_BY_COP,
                3, MPIDatatype<int>, CALL_BY_COP,
                reinterpret_cast<long long>(this), samplePoints.data(),
                &Data.nDim, fValue.data() + i, &nFvalue, workInformation);
  }

  torc_enable_stealing();
  torc_waitall();
  torc_disable_stealing();

  //! Count the function calls
  fc.count();

  //! Print the summary
  std::cout << "server: currentGeneration " << runData.currentGeneration << ": total elapsed time = "
            << "secs, generation elapsed time = "
            << "secs for function calls = " << fc.getLocalFunctionCallsNumber() << std::endl;

  //! Reset the local function counter to zero
  fc.reset();

  //! Check if we should save the mid run data
  if (Data.saveData)
  {
    if (!currentData.save("curgen_db", runData.currentGeneration))
    {
      UMUQFAILRETURN("Failed to write down the current data information!");
    }
  }

  //! Running information for checkpoint restart
  return runData.save();
}

template <typename T, class F>
bool tmcmc<T, F>::iterate()
{
  if (!iterate0())
  {
    UMUQFAILRETURN("Failed to start or restart the algorithm!");
  }

  //! Prepare new generation from the current information
  if (!prepareNewGeneration())
  {
    UMUQFAILRETURN("Failed to prepare the new generation of sample points!");
  }

  //! Broadcast the information to all of the nodes
  runData.broadcast();

  //! Print the sample mean and Sample covariance matrix
  runData.print();

  //! Check the current data probability
  while (runData.generationProbabilty[runData.currentGeneration] < T{1} || runData.currentGeneration < runData.maxGenerations)
  {
    runData.currentGeneration++;
  }
}

template <typename T, class F>
bool tmcmc<T, F>::prepareNewGeneration()
{
  int const nDimSamplePoints = currentData.nDimSamplePoints;
  int const nCurrentSamplePoints = currentData.size();

  //! Total size of the sampling points array Dim * number of sample points
  int const nSize = nCurrentSamplePoints * nDimSamplePoints;

  //! Create an instance of the statistics object
  stats s;

#ifdef DEBUG
  //! Compute vectors of mean and standard deviation for each dimension
  std::vector<T> mean(nDimSamplePoints);
  std::vector<T> stddev(nDimSamplePoints);

  for (int i = 0; i < nDimSamplePoints; i++)
  {
    mean[i] = s.mean<T, T>(currentData.samplePoints.data() + i, nSize, nDimSamplePoints);
    stddev[i] = s.stddev<T, T>(currentData.samplePoints.data() + i, nSize, nDimSamplePoints, mean[i]);
  }

  io f;
  std::cout << "Complete data samples on the current Generation Number = " << runData.currentGeneration << std::endl;
  f.printMatrix<T>("Means", mean.data(), 1, nDimSamplePoints);
  f.printMatrix<T>("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

  //! Now we check to find unique sampling points
  std::vector<T> currentDataUniques(nSize);

  //! Get the uniques samples
  s.unique<T>(currentData.samplePoints, nCurrentSamplePoints, nDimSamplePoints, currentDataUniques);

  //! Set the number of uniques samples
  runData.setUniqueNumber(currentDataUniques.size());

  {
    //! Compute the acceptance rate
    T const acceptanceRate = static_cast<T>(currentDataUniques.size()) / static_cast<T>(nCurrentSamplePoints);

    //! Set the acceptance rate
    runData.setAcceptanceRate(acceptanceRate);
  }

#ifdef DEBUG
  for (int i = 0; i < nDimSamplePoints; i++)
  {
    mean[i] = s.mean<T, T>(currentDataUniques.data() + i, nSize, nDimSamplePoints);
    stddev[i] = s.stddev<T, T>(currentDataUniques.data() + i, nSize, nDimSamplePoints, mean[i]);
  }

  std::cout << "Unique data samples on current Generation Number = " << runData.currentGeneration << std::endl;
  f.printMatrix<T>("Means", mean.data(), 1, nDimSamplePoints);
  f.printMatrix<T>("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

  //! Create database for leaders selection
  leadersData = std::move(database<T>(nDimSamplePoints, Data.eachPopulationSize[runData.currentGeneration]));

  //! Select the new generaion leaders and update the statistics
  if (tStats.selectNewGeneration(Data, currentData, runData, leadersData))
  {
    //! Reset the number of selections for each leader chain
    if (leadersData.resetSelection(Data.minChainLength, Data.maxChainLength))
    {
      //! Update leaders information from current data
      if (leadersData.updateSelection(currentData))
      {
#ifdef DEBUG
        //! Total number of sampling points
        int const nLeadersSamplePoints = static_cast<int>(leadersData.idxPosition);

        //! Total size of the sampling points array Dim * number of sample points
        int const nLeadersSize = nLeadersSamplePoints * nDimSamplePoints;

        for (int i = 0; i < nDimSamplePoints; i++)
        {
          mean[i] = s.mean<T, T>(leadersData.samplePoints.data() + i, nLeadersSize, nDimSamplePoints);
          stddev[i] = s.stddev<T, T>(leadersData.samplePoints.data() + i, nLeadersSize, nDimSamplePoints, mean[i]);
        }

        std::cout << "Leaders samples = " << runData.currentGeneration << std::endl;
        f.printMatrix<T>("Means", mean.data(), 1, nDimSamplePoints);
        f.printMatrix<T>("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

        if (Data.useLocalCovariance)
        {
          UMUQFAILRETURN("Not implemented yet!");
        }

        return true;
      }
      UMUQFAILRETURN("Failed to update information from the current chain to the leaders!");
    }
    UMUQFAILRETURN("Failed to balance the leaders number of chains selection!");
  }
  UMUQFAILRETURN("Failed to select the leaders for the new generation!");
}

template <typename T, class F>
void tmcmcInitTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints, T *Fvalue, int const *nFvalue, int const *WorkInformation)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<T, F> *>(TMCMCObj);

  //! If we have set the fitting function
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
