#ifndef UMUQ_TMCMCSTATS_H
#define UMUQ_TMCMCSTATS_H

#include "data/datatype.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "numerics/multimin.hpp"
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
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold.
 * 
 * \tparam T    Input data type
 * \tparam TOut Output data type (return output result (default is double))
 * 
 * \param  FunValues   An array of log value
 * \param  nFunValues  Number of FunValues 
 * \param  Stride      Element stride 
 * \param  PJ1         \f$ p_{j+1} \f$      
 * \param  pJ          \f$ p_j \f$
 * \param  Tolerance   A prescribed tolerance
 * 
 * \returns the square of the coefficient of variation (COV)
 */
template <typename T, typename TOut = double>
TOut CoefVar(T const *FunValues, int const nFunValues, int const Stride, T const PJ1, T const PJ, T const Tolerance);

template <typename T, typename TOut = double>
TOut CoefVar(T const *FunValues, int const nFunValues, T const PJ1, T const PJ, T const Tolerance);

template <typename T, typename TOut = double>
TOut CoefVar(std::vector<T> const &FunValues, T const PJ1, T const PJ, T const Tolerance);

/*!
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold
 * This function is just a wrapper to call CoefVar function but has the proper Function type that can be used 
 * in multidimensional minimization \sa functionMinimizer
 * 
 * \tparam T Data type 
 * 
 * \param x Input p
 *  
 * \returns The square of the coefficient of variation (COV) for the choice of input p
 */
template <typename T>
T CoefVarFun(T const *x);

/*!
 * \brief Pointer to array of function values
 * 
 * \tparam T 
 */
template <typename T>
T *funValues;

//! Number of function values in funValues array
int nfunValues;

//! Current \f$ p_j \f$
template <typename T>
T pj;

//! Optimizer tolerance
template <typename T>
T tolerance;

/*! \class tmcmcStats
 * \brief 
 * 
 * 
 */
template <typename T>
class tmcmcStats
{
  public:
    /*!
     * \brief Construct a new tmcmc Stats object
     * 
     */
    tmcmcStats();

    /*!
     * \brief Construct a new tmcmc Stats object
     * 
     * \param OptParams optimization Parameters
     */
    explicit tmcmcStats(optimizationParameters<T> const &OptParams);

    /*!
	 * \brief Move constructor, construct a new tmcmcStats object from an input object
     * 
     * \param other  Input tmcmcStats object
	 */
    tmcmcStats(tmcmcStats<T> &&other);

    /*!
	 * \brief Move assignment operator
	 * 
	 * \param other 
	 * \return tmcmcStats<T>& 
	 */
    tmcmcStats<T> &operator=(tmcmcStats<T> &&other);

    /*!
     * \brief Destroy the tmcmc Stats object
     * 
     */
    ~tmcmcStats();

    /*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object \sa psrandom
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<T> *PRNG);

    /*!
     * \brief Find the optimum value of  \f$ p_j \f$
     * 
     * The choice of \f$ p_j : j=1,\cdots,m− 1 \f$ is essential.
     * It is desirable to increase the p values slowly so that the transition between adjacent PDFs 
     * is smooth, but if the increase of the p values is too slow, the required number of 
     * intermediate stages will be huge. More intermediate stages mean more computational cost.
     * \f$ p_{j+1} \f$ should be chosen so that the coefficient of variation tolCOV of the plausibility 
     * weights is equal to a prescribed threshold. 
     * 
     * \param FunValues       Function values
     * \param nFunValues      Number of elements in FunValues
     * \param PJ              \f$ p_j \f$
     * \param OptimumP        Optimum value of p
     * \param OptimumCoefVar  Optimum Coef Var
     * 
     * \return true 
     * \return false If it encounters any problem 
     */
    bool findOptimumP(T const *FunValues, int const nFunValues, T const PJ, T *OptimumP, T *OptimumCoefVar);

    /*!
     * \brief Find the optimum value of  \f$ p_j \f$ through direct search of the \f$ p \in [0, 4] \f$
     * 
     * 
     * \param FunValues       Function values
     * \param nFunValues      Number of elements in FunValues
     * \param PJ              \f$ p_j \f$
     * \param OptimumP        Optimum value of p
     * \param OptimumCoefVar  Optimum Coef Var
     * 
     * \return true 
     * \return false If it encounters any problem 
     */
    bool searchOptimumP(T const *FunValues, int const nFunValues, T const PJ, T *OptimumP, T *OptimumCoefVar);

    bool selectNewGeneration(stdata<T> &StreamData, database<T> &CurrentData, runinfo<T> &RunData, database<T> &Leaders);

  private:
    // Make it noncopyable
    tmcmcStats(tmcmcStats<T> const &) = delete;

    // Make it not assignable
    tmcmcStats<T> &operator=(tmcmcStats<T> const &) = delete;

  private:
    //! Optimizer information
    optimizationParameters<T> optParams;

    //! Create an instance of the function minimizer
    std::unique_ptr<umuq::functionMinimizer<T>> fMinimizer;

    //! pseudo-random numbers
    psrandom<T> *prng;
};

template <typename T>
tmcmcStats<T>::tmcmcStats() : optParams(),
                              fMinimizer(new umuq::simplexNM2<T>),
                              prng(nullptr)
{
    //! First we have to set the minimizer dimension
    if (!fMinimizer->reset(1))
    {
        UMUQFAIL("Failed to set the minimizer dimension!");
    }

    //! Set the prescribed tolerance
    tolerance<T> = optParams.Tolerance;
}

template <typename T>
tmcmcStats<T>::tmcmcStats(optimizationParameters<T> const &OptParams) : optParams(OptParams),
                                                                        prng(nullptr)
{
    //! Get the correct instance of the minimizer
    switch (optParams.FunctionMinimizerType)
    {
    case (FunctionMinimizerTypes::SIMPLEXNM):
        fMinimizer.reset(new umuq::simplexNM<T>);
        break;
    case (FunctionMinimizerTypes::SIMPLEXNM2):
        fMinimizer.reset(new umuq::simplexNM2<T>);
        break;
    case (FunctionMinimizerTypes::SIMPLEXNM2RND):
        fMinimizer.reset(new umuq::simplexNM2Rnd<T>);
        break;
    default:
        UMUQFAIL("Unknown function minimizer type!");
        break;
    }

    //! First we have to set the minimizer dimension
    if (!fMinimizer->reset(1))
    {
        UMUQFAIL("Failed to set the minimizer dimension!");
    }

    //! set the prescribed tolerance
    tolerance<T> = optParams.Tolerance;
}

template <typename T>
tmcmcStats<T>::tmcmcStats(tmcmcStats<T> &&other) : optParams(std::move(other.optParams)),
                                                   fMinimizer(std::move(other.fMinimizer))
{
    prng = other.prng;
}

template <typename T>
tmcmcStats<T> &tmcmcStats<T>::operator=(tmcmcStats<T> &&other)
{
    optParams = std::move(other.optParams);
    fMinimizer = std::move(other.fMinimizer);
    prng = other.prng;
    return *this;
}

template <typename T>
tmcmcStats<T>::~tmcmcStats(){};

template <typename T>
inline bool tmcmcStats<T>::setRandomGenerator(psrandom<T> *PRNG)
{
    if (PRNG)
    {
        if (PRNG_initialized)
        {
            prng = PRNG;
            return true;
        }
        UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to this distribution!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename T>
bool tmcmcStats<T>::findOptimumP(T const *FunValues, int const nFunValues, T const PJ, T *OptimumP, T *OptimumCoefVar)
{
    //! Set the helper pointer & variables
    funValues<T> = FunValues;
    nfunValues = nFunValues;
    pj<T> = PJ;

    //! Second, we have to set the function, input vector and stepsize
    if (!fMinimizer->set(CoefVarFun<T>, &pj<T>, &optParams.Step))
    {
        UMUQFAILRETURN("Failed to set the minimizer!");
    }

    //! Third, initilize the minimizer
    if (!fMinimizer->init())
    {
        UMUQFAILRETURN("Failed to initialize the minimizer!");
    }

    if (optParams.Display)
    {
        T *x = fMinimizer->getX();
        std::cout << "x =" << x[0] << std::endl;
    }

    //! Forth, iterate until we reach the absolute tolerance of 1e-3

    //! Fail:-1, Success:0, Continue:1
    int status = 1;
    int iter = 0;

    T const tol = optParams.Tolerance;

    while (iter < optParams.MaxIter && status == 1)
    {
        iter++;

        if (!fMinimizer->iterate())
        {
            UMUQFAILRETURN("Failed to iterate the minimizer!");
        }

        if (optParams.Display)
        {
            T *x = fMinimizer->getX();
            std::cout << iter << ": ";
            std::cout << "CoefVar(p=" << x[0] << ") =" << fMinimizer->getMin() << ", & characteristic size =" << fMinimizer->getSize() << std::endl;
        }

        status = fMinimizer->testSize(tol);
    }

    if (status == 0 || status == 1)
    {
        OptimumP = fMinimizer->getX();
        *OptimumCoefVar = fMinimizer->getMin();

        if (optParams.Display)
        {
            std::cout << fMinimizer->getName() << ", on CoefVarFun : " << iter << " iters, CoefVar(p)=" << *OptimumCoefVar << std::endl;
            std::cout << ((status == 0) ? "Converged to minimum at p = " : "Stopped at p = ");
            std::cout << *OptimumP << std::endl;
        }
        return (status ? (std::abs(fMinimizer->getMin()) <= tol) : true);
    }
    return false;
}

template <typename T>
bool tmcmcStats<T>::searchOptimumP(T const *FunValues, int const nFunValues, T const PJ, T *OptimumP, T *OptimumCoefVar)
{
    T MinValp = T{};
    T MaxValp = T{4};

    //! Initilize the step size to the prescribed user value
    T stepSize(optParams.Step);

    for (;;)
    {
        if (optParams.Display)
        {
            std::cout << "Search for optimum p in [" << MinValp << ", " << MaxValp << "] with step size =" << stepSize << std::endl;
        }

        //! Define some variables to use in the search
        bool found(false);

        //! Maximum number of iterations
        std::size_t const MaxIter = static_cast<std::size_t>((MaxValp - MinValp) / stepSize);

        //! Optimum value of p
        T optimumP(MinValp);

        //! Max value of CoefVar
        T MaxCoefVar(std::numeric_limits<T>::max());

        //! We search the space for the optimum value
        for (std::size_t iter = 0; iter < MaxIter; iter++)
        {
            T const p = MinValp + iter * stepSize;
            T const NewCoefVar = CoefVarFun<T>(&p);

            if (NewCoefVar < MaxCoefVar)
            {
                MaxCoefVar = NewCoefVar;
                optimumP = p;

                if (MaxCoefVar <= tolerance<T>)
                {
                    found = true;
                    break;
                }
            }
        }

        //! Success:0
        if (found)
        {
            *OptimumP = optimumP;
            *OptimumCoefVar = MaxCoefVar;

            if (optParams.Display)
            {
                std::cout << "Serach on CoefVarFun : CoefVar(p)=" << *OptimumCoefVar << std::endl;
                std::cout << "Converged to minimum at p = " << *OptimumP << std::endl;
            }
            return true;
        }
        else
        {
            //! If optimumP is not within Tolerance, we can go back and retry with better refinement (more iterations)

            MinValp = optimumP - 10 * stepSize;
            if (MinValp < 0)
            {
                MinValp = T{};
            }

            MaxValp = optimumP + 10 * stepSize;
            if (MaxValp > T{4})
            {
                MaxValp = T{4};
            }

            stepSize /= T{10};
            if (stepSize < machinePrecision<T>)
            {
                //! Fail:-1
                UMUQFAILRETURN("Could not find the optimum value of p through search!");
            }
        }
    }
    UMUQFAILRETURN("Could not find the optimum value of p through search!");
}

template <typename T>
bool tmcmcStats<T>::selectNewGeneration(stdata<T> &StreamData, database<T> &CurrentData, runinfo<T> &RunData, database<T> &Leaders)
{
    //! Initilize the step size to the prescribed user value
    T const stepSize(optParams.Step);

    //! current generation
    int const currentGeneration = RunData.currentGeneration;
    int const newGeneration = currentGeneration + 1;

    //! Get the total number of samples at the current generation
    int const nSamples = StreamData.eachPopulationSize[currentGeneration];

    //! Get the pointerto the CoefVar
    T *CoefVar = RunData.CoefVar.data();

    //! Probabilty at each generation
    T *generationProbabilty = RunData.generationProbabilty.data();

    //! Probability of the current generation
    T const PJ = generationProbabilty[currentGeneration];

    //!
    T *logselection = RunData.logselection.data();

    //! Get the pointer to the function values
    T *Fvalue = CurrentData.Fvalue.get();

    //! Total number of function values
    int const nFvalue = CurrentData.getSize();

    T optimumP(T{});
    T optimumCoefVar(T{});

    bool status(true);

    if (!findOptimumP(Fvalue, nFvalue, PJ, &optimumP, &optimumCoefVar))
    {
        status = searchOptimumP(Fvalue, nFvalue, PJ, &optimumP, &optimumCoefVar);
    }

    if (status && optimumP > PJ)
    {
        generationProbabilty[newGeneration] = optimumP;
        CoefVar[newGeneration] = optimumCoefVar;
    }
    else
    {
        generationProbabilty[newGeneration] = PJ + 0.1 * stepSize;
        CoefVar[newGeneration] = CoefVar[currentGeneration];
    }

    if (generationProbabilty[newGeneration] > 1)
    {
        generationProbabilty[newGeneration] = T{1};
    }

    std::vector<T> weight(nFvalue);

    {
        T const pdiff = generationProbabilty[newGeneration] - generationProbabilty[currentGeneration];

        for (int i = 0; i < nFvalue; i++)
        {
            weight[i] = Fvalue[i] * pdiff;
        }
    }

    stats s;
    T const weightMax = s.maxelement<T, T>(weight);

    std::for_each(weight.begin(), weight.end(), [&](T &w_i) { w_i = std::exp(w_i - weightMax); });

    if (optParams.Display)
    {
        io f;
        f.printMatrix<T>("Weight matrix", weight.data(), 1, nFvalue);
    }

    //! Compute the weight sum
    T const weightSum = s.sum<T, T>(weight);

    //! Normalize the weight
    std::for_each(weight.begin(), weight.end(), [&](T &w_i) { w_i /= weightSum; });

    if (optParams.Display)
    {
        io f;
        f.printMatrix<T>("p matrix", weight.data(), 1, nFvalue);
    }

    logselection[currentGeneration] = std::log(weightSum) + weightMax - std::log(static_cast<T>(nFvalue));

    if (optParams.Display)
    {
        io f;
        f.printMatrix<T>("log selection matrix", logselection, 1, newGeneration);
    }

    //! Compute the average of weight
    T const weightMean = s.mean<T, T>(weight);

    //! Compute the standard deviation of weight
    T const weightStddev = s.stddev<T, T>(weight, weightMean);

    //! Compute the current Generation CoefVar
    CoefVar[currentGeneration] = weightStddev / weightMean;

    if (optParams.Display)
    {
        io f;
        f.printMatrix<T>("CoefVar matrix", logselection, 1, newGeneration);
    }

    int *nSelection = Leaders.nSelection.get();

    if (prng)
    {
        //!
        prng->multinomial(weight.data(), nFvalue, nSamples, nSelection);

        if (optParams.Display)
        {
            io f;
            f.printMatrix<int>("Number of selection = [", nSelection, 1, nFvalue, "]\n----------------------------------------\n");
        }

        //! Compute vectors of mean and standard deviation for each dimension
        std::vector<T> thetaMean(CurrentData.ndimParray);
        std::vector<T> thetaStddev(CurrentData.ndimParray);
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename T, typename TOut>
TOut CoefVar(T const *FunValues, int const nFunValues, int const Stride, T const PJ1, T const PJ, T const Tolerance)
{
    arrayWrapper<T> iArray(FunValues, nFunValues, Stride);

    std::vector<TOut> weight(iArray.size());

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        T const fMaxValue = s.maxelement<T>(iArray);

        TOut const diff = static_cast<TOut>(PJ1 - PJ);

        //Compute the weight
        for (std::size_t i = 0; i < iArray.size(); i++)
        {
            weight[i] = std::exp((iArray[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        TOut const weightsum = s.sum<TOut, TOut>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](TOut &w) { w /= weightsum; });
    }

    //Compute the mean
    TOut const weightmean = s.mean<TOut, TOut>(weight);

    //Compute the standard deviation
    TOut const weightstddev = s.stddev<TOut, TOut>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename T, typename TOut>
TOut CoefVar(T const *FunValues, int const nFunValues, T const PJ1, T const PJ, T const Tolerance)
{
    std::vector<TOut> weight(nFunValues);

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        T const fMaxValue = s.maxelement<T>(FunValues, nFunValues);

        TOut const diff = static_cast<TOut>(PJ1 - PJ);

        //Compute the weight
        for (int i = 0; i < nFunValues; i++)
        {
            weight[i] = std::exp((FunValues[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        TOut const weightsum = s.sum<TOut, TOut>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](TOut &w) { w /= weightsum; });
    }

    //Compute the mean
    TOut const weightmean = s.mean<TOut, TOut>(weight);

    //Compute the standard deviation
    TOut const weightstddev = s.stddev<TOut, TOut>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename T, typename TOut>
TOut CoefVar(std::vector<T> const &FunValues, T const PJ1, T const PJ, T const Tolerance)
{
    std::vector<TOut> weight(FunValues.size());

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        T const fMaxValue = s.maxelement<T>(FunValues);

        TOut const diff = static_cast<TOut>(PJ1 - PJ);

        //Compute the weight
        for (std::size_t i = 0; i < FunValues.size(); i++)
        {
            weight[i] = std::exp((FunValues[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        TOut const weightsum = s.sum<TOut, TOut>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](TOut &w) { w /= weightsum; });
    }

    //Compute the mean
    TOut const weightmean = s.mean<TOut, TOut>(weight);

    //Compute the standard deviation
    TOut const weightstddev = s.stddev<TOut, TOut>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename T>
T CoefVarFun(T const *x)
{
    return CoefVar<T, T>(funValues<T>, nfunValues, *x, pj<T>, tolerance<T>);
}

} // namespace tmcmc
} // namespace umuq

#endif // UMUQ_TMCMCSTATS
