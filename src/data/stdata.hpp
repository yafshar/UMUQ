#ifndef UMUQ_STDATA_H
#define UMUQ_STDATA_H

#include "core/core.hpp"
#include "misc/parser.hpp"
#include "io/io.hpp"

namespace umuq
{

namespace tmcmc
{

/*! \class optimizationParameters
 * \ingroup TMCMC_Module
 * 
 * \brief This is a class to set the optimization parameters 
 * 
 * \tparam T Data type
 */
template <typename T>
struct optimizationParameters
{
    /*!
     *  \brief Default constructor for the default variables
     *    
     */
    optimizationParameters();

    /*!
     * \brief Construct a new optimization Parameters object
     * 
     * \param other 
     */
    explicit optimizationParameters(optimizationParameters<T> const &other);

    /*!
     * \brief Move constructor, construct a new optimizationParameters object from an input object
     * 
     * \param other  Input optimizationParameters object
     */
    optimizationParameters(optimizationParameters<T> &&other);

    /*!
     * \brief Copy constructor, construct a new optimizationParameters object from an input object
     * 
     * \param other 
     * \returns optimizationParameters<T>& 
     */
    optimizationParameters<T> &operator=(optimizationParameters<T> const &other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other 
     * \returns optimizationParameters<T>& 
     */
    optimizationParameters<T> &operator=(optimizationParameters<T> &&other);

    /*!
     * \brief Destroy the optimization Parameters object
     * 
     */
    ~optimizationParameters();

    /*!
     * \brief Reset the optimization parameters which is used in the function optimizer
     * 
     * The default values are:
     * - \b maxIter = 100
     * - \b display = 0 (Off)
     * - \b functionMinimizerType = 2 (SIMPLEXNM2, The Simplex method of Nelder and Mead)
     * - \b tolerance = 1e-6
     * - \b step = 1e-5
     * 
     * \returns true 
     * \returns false 
     */
    void reset();
    void reset(int const maxIter, int const display, int const functionMinimizerType, T const tolerance, T const step);

    //! Maximum number of iterations
    int MaxIter;
    //! Debugging flag, if the minimization steps are shown or not
    int Display;
    //! function minimizer type (per default it is simplex2)
    int FunctionMinimizerType;
    //! minimizer tolerance
    T Tolerance;
    //! Minimizer step size
    T Step;
};

/*! \class stdata
 * \ingroup TMCMC_Module
 * 
 * \brief stream data type class
 *
 * - \b nDim                       Problem Dimension
 * - \b maxGenerations             Maximum number of generations
 * - \b populationSize             Sampling population size
 * - \b lastPopulationSize         Sampling population size in the final generation
 * - \b auxilSize                  Auxillary data size
 * - \b minChainLength             Minimum size of the chain in the TMCMC algorithm (default 1)
 * - \b maxChainLength             Maximum size of the chain in the TMCMC algorithm (default 1)
 * - \b seed                       Random number initial seed
 * - \b samplingType               Sampling type which is : 0: uniform, 1: gaussian, 2: file
 * - \b priorType                  Prior type which is :   0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4: composit
 * - \b iPlot                      1 for printing the data and 0 for not
 * - \b saveData                   1 for saving the data and 0 for not
 * - \b useCmaProposal             Indicator if we use the CMA proposal or not
 * - \b useLocalCovariance         Indicator if we use the local covariance or not
 * - \b lb                         Generic lower bound (It is -6 per default)
 * - \b ub                         Generic upper bound (It is 6 per default)
 * - \b coefVarPresetThreshold     A preset threshold
 * - \b bbeta                      \f$ \beta \f$ parameter in the TMCMC algorithm
 * - \b options                    Optimization parameter
 * - \b eachPopulationSize         Sampling population size for each generation
 * - \b lowerBound                 Sampling domain lower bounds for each dimension
 * - \b upperBound                 Sampling domain upper bounds for each dimension
 * - \b compositePriorDistribution Composite distribution as a prior
 * - \b priorMu                    Prior mean, in case of gamma distribution it is alpha
 * - \b priorSigma                 Prior standard deviation 
 * - \b auxilData                  Auxillary data
 */
template <typename T>
class stdata
{
  public:
    /*!
     * \brief Default constructor
     *    
     */
    stdata();

    /*!
     * \brief constructor for the default input variables
     *    
     */
    stdata(int probdim, int MaxGenerations, int PopulationSize);

    /*!
     * \brief Default destructor 
     *    
     */
    ~stdata() {}

    /*!
     * \brief Move constructor, construct a new stdata object from an input object
     * 
     * \param other  Input stdata object
     */
    stdata(stdata<T> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other 
     * \returns stdata<T>& 
     */
    stdata<T> &operator=(stdata<T> &&other);

    /*!
     * \brief reset the stream data values to the input values
     * 
     * \param probdim          Problem Dimension
     * \param MaxGenerations   Maximum number of generations
     * \param PopulationSize   Sampling population size
     * 
     * \returns true 
     * \returns false If there is not enough memory available for allocating the data
     */
    bool reset(int probdim, int MaxGenerations, int PopulationSize);

    /*!
     * \brief load the input file fname
     *
     * \param fname              name of the input file
     * \returns true on success
     */
    bool load(const char *fname = "tmcmc.par");
    bool load(std::string const &fname = "tmcmc.par");

    /*!
     * \brief Swap the stdata objects
     * 
     * \param other 
     */
    void swap(stdata<T> &other);

  private:
    /*!
     * \brief Delete a stdata object copy construction
     * 
     * Make it noncopyable.
     */
    stdata(stdata<T> const &) = delete;

    /*!
     * \brief Delete a stdata object assignment
     * 
     * Make it nonassignable
     * 
     * \returns stdata<T>& 
     */
    stdata<T> &operator=(stdata<T> const &) = delete;

  public:
    //! Problem Dimension
    int nDim;

    //! Maximum number of generations
    int maxGenerations;

    //! Sampling population size
    int populationSize;

    //! Sampling population size in the final generation
    int lastPopulationSize;

    //! Auxillary data size
    int auxilSize;

    //! Minimum size of the chain in the TMCMC algorithm (default 1)
    int minChainLength;

    //! Maximum size of the chain in the TMCMC algorithm (default 1)
    int maxChainLength;

    //! Random number initial seed
    long seed;

    //! Sampling type which is : 0: uniform, 1: gaussian, 2: file
    int samplingType;

    //! Prior type which is :   0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite
    int priorType;

    //! 1 for printing the data and 0 for not
    int iPlot;

    //! 1 for saving the data and 0 for not
    int saveData;

    //! Indicator if we use the CMA proposal or not
    int useCmaProposal;

    //! Indicator if we use the local covariance or not
    int useLocalCovariance;

  private:
    //! Generic lower bound (It is -6 per default)
    T lb;

    //! Generic upper bound (It is 6 per default)
    T ub;

  public:
    /*!
     * A preset threshold for coefficient of variation of the plausibility of weights.<br>
     * At each stage \f$ j, \f$ of the MCMC algorithm, \f$ \zeta_{j+1} \f$ is chosen such that 
     * the coefficient of variation of \f$ w_{j,k} \f$ is smaller than some preset threshold. 
     * 
     * Reference: <br>
     * Wu S, et. al. "Bayesian Annealed Sequential Importance Sampling: An Unbiased Version 
     * of Transitional Markov Chain Monte Carlo." ASME J. Risk Uncertainty Part B. 2017;4(1)
     */
    T coefVarPresetThreshold;

    /*!
     * \f$ \beta, \f$ a user-specified scaling factor in the TMCMC algorithm.<br>
     * The proposal PDF for the MCMC step is a Gaussian distribution centered at the sample with covariance equal to 
     * \f$ \beta^2 COV(\Theta(j)), \f$ where \f$ \beta, \f$ is a user-specified scaling factor, and  
     * \f$ \Theta(j) \f$ is the collective samples from MCMC step.
     */
    T bbeta;

    //! Optimization parameter
    optimizationParameters<T> options;

  public:
    //! Sampling population size for each generation
    std::vector<int> eachPopulationSize;

    //! Sampling domain lower bounds for each dimension
    std::vector<T> lowerBound;

    //! Sampling domain upper bounds for each dimension
    std::vector<T> upperBound;

    //! Composite distribution as a prior
    std::vector<int> compositePriorDistribution;

    //! Prior parameter 1
    std::vector<T> priorParam1;

    //! Prior parameter 2
    std::vector<T> priorParam2;

    //! Auxillary data
    std::vector<T> auxilData;
};

template <typename T>
stdata<T>::stdata() : nDim(0),
                      maxGenerations(0),
                      populationSize(0),
                      lastPopulationSize(0),
                      auxilSize(0),
                      minChainLength(1),
                      maxChainLength(1),
                      seed(280675),
                      samplingType(0),
                      priorType(0),
                      iPlot(0),
                      saveData(1),
                      useCmaProposal(0),
                      useLocalCovariance(0),
                      lb(-static_cast<T>(6)),
                      ub(static_cast<T>(6)),
                      coefVarPresetThreshold(static_cast<T>(1)),
                      bbeta(static_cast<T>(0.2)),
                      options(){};

template <typename T>
stdata<T>::stdata(int probdim, int MaxGenerations, int PopulationSize) : nDim(probdim),
                                                                         maxGenerations(MaxGenerations),
                                                                         populationSize(PopulationSize),
                                                                         lastPopulationSize(PopulationSize),
                                                                         auxilSize(0),
                                                                         minChainLength(1),
                                                                         maxChainLength(1),
                                                                         seed(280675),
                                                                         samplingType(0),
                                                                         priorType(0),
                                                                         iPlot(0),
                                                                         saveData(1),
                                                                         useCmaProposal(0),
                                                                         useLocalCovariance(0),
                                                                         lb(-static_cast<T>(6)),
                                                                         ub(static_cast<T>(6)),
                                                                         coefVarPresetThreshold(static_cast<T>(1)),
                                                                         bbeta(static_cast<T>(0.2)),
                                                                         options(),
                                                                         eachPopulationSize(maxGenerations),
                                                                         lowerBound(nDim, T{}),
                                                                         upperBound(nDim, T{}),
                                                                         priorParam1(nDim, T{}),
                                                                         priorParam2(nDim * nDim, T{})
{
    for (int i = 0, k = 0; i < nDim; i++)
    {
        for (int j = 0; j < nDim; j++, k++)
        {
            if (i == j)
            {
                priorParam2[k] = static_cast<T>(1);
            }
        }
    }

    std::fill(eachPopulationSize.begin(), eachPopulationSize.end(), populationSize);
}

template <typename T>
stdata<T>::stdata(stdata<T> &&other)
{
    nDim = other.nDim;
    maxGenerations = other.maxGenerations;
    populationSize = other.populationSize;
    lastPopulationSize = other.lastPopulationSize;
    auxilSize = other.auxilData;
    minChainLength = other.minChainLength;
    maxChainLength = other.maxChainLength;
    seed = other.seed;
    samplingType = other.samplingType;
    priorType = other.priorType;
    iPlot = other.iPlot;
    saveData = other.saveData;
    useCmaProposal = other.useCmaProposal;
    useLocalCovariance = other.useLocalCovariance;
    lb = other.lb;
    ub = other.ub;
    coefVarPresetThreshold = other.coefVarPresetThreshold;
    bbeta = other.bbeta;
    options = std::move(other.options);
    eachPopulationSize = std::move(other.eachPopulationSize);
    lowerBound = std::move(other.lowerBound);
    upperBound = std::move(other.upperBound);
    compositePriorDistribution = std::move(other.compositePriorDistribution);
    priorParam1 = std::move(other.priorParam1);
    priorParam2 = std::move(other.priorParam2);
    auxilData = std::move(other.auxilData);
}

template <typename T>
stdata<T> &stdata<T>::operator=(stdata<T> &&other)
{
    nDim = other.nDim;
    maxGenerations = other.maxGenerations;
    populationSize = other.populationSize;
    lastPopulationSize = other.lastPopulationSize;
    auxilSize = other.auxilData;
    minChainLength = other.minChainLength;
    maxChainLength = other.maxChainLength;
    seed = other.seed;
    samplingType = other.samplingType;
    priorType = other.priorType;
    iPlot = other.iPlot;
    saveData = other.saveData;
    useCmaProposal = other.useCmaProposal;
    useLocalCovariance = other.useLocalCovariance;
    lb = other.lb;
    ub = other.ub;
    coefVarPresetThreshold = other.coefVarPresetThreshold;
    bbeta = other.bbeta;
    options = std::move(other.options);
    eachPopulationSize = std::move(other.eachPopulationSize);
    lowerBound = std::move(other.lowerBound);
    upperBound = std::move(other.upperBound);
    compositePriorDistribution = std::move(other.compositePriorDistribution);
    priorParam1 = std::move(other.priorParam1);
    priorParam2 = std::move(other.priorParam2);
    auxilData = std::move(other.auxilData);

    return *this;
}

template <typename T>
void stdata<T>::swap(stdata<T> &other)
{
    std::swap(nDim, other.nDim);
    std::swap(maxGenerations, other.maxGenerations);
    std::swap(populationSize, other.populationSize);
    std::swap(lastPopulationSize, other.lastPopulationSize);
    std::swap(auxilSize, other.auxilData);
    std::swap(minChainLength, other.minChainLength);
    std::swap(maxChainLength, other.maxChainLength);
    std::swap(seed, other.seed);
    std::swap(samplingType, other.samplingType);
    std::swap(priorType, other.priorType);
    std::swap(iPlot, other.iPlot);
    std::swap(saveData, other.saveData);
    std::swap(useCmaProposal, other.useCmaProposal);
    std::swap(useLocalCovariance, other.useLocalCovariance);
    std::swap(lb, other.lb);
    std::swap(ub, other.ub);
    std::swap(coefVarPresetThreshold, other.coefVarPresetThreshold);
    std::swap(bbeta, other.bbeta);
    std::swap(options.Display, other.options.Display);
    std::swap(options.MaxIter, other.options.MaxIter);
    std::swap(options.FunctionMinimizerType, other.options.FunctionMinimizerType);
    std::swap(options.Step, other.options.Step);
    std::swap(options.Tolerance, other.options.Tolerance);
    eachPopulationSize.swap(other.eachPopulationSize);
    lowerBound.swap(other.lowerBound);
    upperBound.swap(other.upperBound);
    compositePriorDistribution.swap(other.compositePriorDistribution);
    priorParam1.swap(other.priorParam1);
    priorParam2.swap(other.priorParam2);
    auxilData.swap(other.auxilData);
}

template <typename T>
bool stdata<T>::reset(int probdim, int MaxGenerations, int PopulationSize)
{
    auxilSize = 0;
    minChainLength = 1;
    maxChainLength = 1;
    seed = 280675;
    samplingType = 0;
    priorType = 0;
    iPlot = 0;
    saveData = 1;
    useCmaProposal = 0;
    useLocalCovariance = 0;
    lb = -static_cast<T>(6);
    ub = static_cast<T>(6);
    coefVarPresetThreshold = static_cast<T>(1);
    bbeta = static_cast<T>(0.2);
    options.reset();

    if (probdim == 0 || MaxGenerations == 0 || PopulationSize == 0)
    {
        nDim = 0;
        maxGenerations = 0;
        populationSize = 0;
        lastPopulationSize = 0;

        eachPopulationSize.resize(0);
        lowerBound.resize(0);
        upperBound.resize(0);
        priorParam1.resize(0);
        priorParam2.resize(0);

        UMUQWARNING("Reseting to size zero!");

        return true;
    }

    nDim = probdim;
    maxGenerations = MaxGenerations;
    populationSize = PopulationSize;
    lastPopulationSize = PopulationSize;

    try
    {
        eachPopulationSize.resize(maxGenerations);
        lowerBound.resize(nDim, T{});
        upperBound.resize(nDim, T{});
        priorParam1.resize(nDim, T{});
        priorParam2.resize(nDim * nDim, T{});
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }

    for (int i = 0, k = 0; i < nDim; i++)
    {
        for (int j = 0; j < nDim; j++, k++)
        {
            if (i == j)
            {
                priorParam2[k] = static_cast<T>(1);
            }
        }
    }

    std::fill(eachPopulationSize.begin(), eachPopulationSize.end(), populationSize);

    return true;
}

/*!
 * \brief load the input file fname for setting the input variables
 * 
 * \tparam T      Data type
 * 
 * \param fname   Input file name
 *  
 * \returns true 
 * \returns false 
 */
template <typename T>
bool stdata<T>::load(const char *fname)
{
    // We use an IO object to open and read a file
    umuq::io f;
    if (f.isFileExist(fname))
    {
        if (f.openFile(fname, f.in))
        {
            // We need a parser object to parse
            umuq::parser p;

            // These are temporary variables
            int probdim = nDim;
            int maxgens = maxGenerations;
            int datanum = populationSize;

            // Read each line in the file and skip all the commented and empty line with the default comment "#"
            while (f.readLine())
            {
                // Parse the line into line arguments
                p.parse(f.getLine());

                if (p.at<std::string>(0) == "nDim")
                {
                    nDim = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "maxGenerations")
                {
                    maxGenerations = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "populationSize")
                {
                    populationSize = p.at<int>(1);
                }
            }

            {
                bool isInitialized = !(probdim == nDim && maxgens == maxGenerations && datanum == populationSize);
                if (isInitialized)
                {
                    if (!reset(nDim, maxGenerations, populationSize))
                    {
                        return false;
                    }
                }
            }

            f.rewindFile();

            // Read each line in the file and skip all the commented and empty line with the default comment "#"
            while (f.readLine())
            {
                // Parse the line into line arguments
                p.parse(f.getLine());

                if (p.at<std::string>(0) == "coefVarPresetThreshold")
                {
                    coefVarPresetThreshold = p.at<T>(1);
                }
                else if (p.at<std::string>(0) == "bbeta")
                {
                    bbeta = p.at<T>(1);
                }
                else if (p.at<std::string>(0) == "seed")
                {
                    seed = p.at<long>(1);
                }
                else if (p.at<std::string>(0) == "opt.MaxIter")
                {
                    options.MaxIter = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "opt.Tol")
                {
                    options.Tolerance = p.at<T>(1);
                }
                else if (p.at<std::string>(0) == "opt.FMin" || p.at<std::string>(0) == "opt.Minimizer")
                {
                    options.FunctionMinimizerType = p.at<T>(1);
                }
                else if (p.at<std::string>(0) == "opt.Display")
                {
                    options.Display = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "opt.Step")
                {
                    options.Step = p.at<T>(1);
                }
                else if (p.at<std::string>(0) == "priorType")
                {
                    priorType = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "iPlot")
                {
                    iPlot = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "saveData")
                {
                    saveData = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "Bdef")
                {
                    lb = p.at<T>(1);
                    ub = p.at<T>(2);
                }
                else if (p.at<std::string>(0) == "minChainLength")
                {
                    minChainLength = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "maxChainLength")
                {
                    maxChainLength = p.at<int>(1);
                }
                else if (p.at<std::string>(0) == "useLocalCovariance")
                {
                    useLocalCovariance = p.at<int>(1);
                }
            }

            int n = nDim;
            int found;
            while (n--)
            {
                f.rewindFile();

                found = 0;
                std::string strTemp("B" + std::to_string(n));

                while (f.readLine())
                {
                    p.parse(f.getLine());

                    if (p.at<std::string>(0) == strTemp)
                    {
                        lowerBound[n] = p.at<T>(1);
                        upperBound[n] = p.at<T>(2);
                        found = 1;
                        break;
                    }
                }

                // In case we do not find the value, we use the default lower bound and upper bound
                if (!found)
                {
                    lowerBound[n] = lb;
                    upperBound[n] = ub;
                }
            }

            // 0: uniform
            if (priorType == 0)
            {
                for (n = 0; n < nDim; n++)
                {
                    priorParam1[n] = lowerBound[n];
                    priorParam2[n] = upperBound[n];
                }
            }

            // 1: gaussian
            if (priorType == 1)
            {
                f.rewindFile();

                while (f.readLine())
                {
                    p.parse(f.getLine());

                    if (p.at<std::string>(0) == "priorMu")
                    {
                        for (n = 0; n < nDim; n++)
                        {
                            priorParam1[n] = p.at<T>(n + 1);
                        }
                        break;
                    }
                }

                f.rewindFile();

                while (f.readLine())
                {
                    p.parse(f.getLine());

                    if (p.at<std::string>(0) == "priorSigma")
                    {
                        for (n = 0; n < nDim * nDim; n++)
                        {
                            priorParam2[n] = p.at<T>(n + 1);
                        }
                        break;
                    }
                }
            }

            // 2: exponential
            if (priorType == 2)
            {
                f.rewindFile();

                while (f.readLine())
                {
                    p.parse(f.getLine());

                    if (p.at<std::string>(0) == "priorMu")
                    {
                        for (n = 0; n < nDim; n++)
                        {
                            priorParam1[n] = p.at<T>(n + 1);
                        }
                        break;
                    }
                }
            }

            // 3: gamma
            if (priorType == 3)
            {
                f.rewindFile();

                while (f.readLine())
                {
                    p.parse(f.getLine());

                    // \f$ \alpha \f$ parameter in Gamma distribution
                    if (p.at<std::string>(0) == "priorGammaAlpha")
                    {
                        for (n = 0; n < nDim; n++)
                        {
                            priorParam1[n] = p.at<T>(n + 1);
                        }
                        break;
                    }
                }

                f.rewindFile();

                while (f.readLine())
                {
                    p.parse(f.getLine());

                    // \f$ \beta \f$ parameter in Gamma distribution
                    if (p.at<std::string>(0) == "priorGammaBeta")
                    {
                        for (n = 0; n < nDim; n++)
                        {
                            priorParam2[n] = p.at<T>(n + 1);
                        }
                        break;
                    }
                }
            }

            // 4:composite
            if (priorType == 4)
            {
                try
                {
                    compositePriorDistribution.resize(nDim, 0);
                }
                catch (...)
                {
                    UMUQFAILRETURN("Failed to allocate memory!");
                }

                n = nDim;
                while (n--)
                {
                    f.rewindFile();

                    found = 0;
                    std::string strTemp("C" + std::to_string(n));

                    while (f.readLine())
                    {
                        p.parse(f.getLine());

                        if (p.at<std::string>(0) == strTemp)
                        {
                            compositePriorDistribution[n] = p.at<int>(1);
                            priorParam1[n] = p.at<T>(2);
                            if (compositePriorDistribution[n] != 2)
                            {
                                priorParam2[n] = p.at<T>(3);
                            }
                            break;
                        }
                    }
                }
            }

            /* new, parse auxilSize and auxilData */
            f.rewindFile();

            while (f.readLine())
            {
                p.parse(f.getLine());

                if (p.at<std::string>(0) == "auxilSize")
                {
                    auxilSize = p.at<int>(1);
                    break;
                }
            }

            if (auxilSize > 0)
            {
                try
                {
                    auxilData.resize(auxilSize);
                }
                catch (...)
                {
                    UMUQFAILRETURN("Failed to allocate memory!");
                }

                f.rewindFile();

                while (f.readLine())
                {
                    p.parse(f.getLine());

                    if (p.at<std::string>(0) == "auxilData")
                    {
                        for (n = 0; n < auxilSize; n++)
                        {
                            auxilData[n] = p.at<T>(n + 1);
                        }
                        break;
                    }
                }
            }

            f.closeFile();

            return true;
        }
        UMUQFAILRETURN("An error has occurred on the associated stream from opening the file!");
    }
    UMUQFAILRETURN("Requested File does not exist in the current PATH!!");
}

template <typename T>
bool stdata<T>::load(std::string const &fname)
{
    return load(&fname[0]);
}

template <typename T>
optimizationParameters<T>::optimizationParameters() : MaxIter(100),
                                                      Display(0),
                                                      FunctionMinimizerType(2),
                                                      Tolerance(1e-6),
                                                      Step(1e-5){};

template <typename T>
optimizationParameters<T>::optimizationParameters(optimizationParameters<T> const &other)
{
    Display = other.Display;
    MaxIter = other.MaxIter;
    FunctionMinimizerType = other.FunctionMinimizerType;
    Step = other.Step;
    Tolerance = other.Tolerance;
}

template <typename T>
optimizationParameters<T>::optimizationParameters(optimizationParameters<T> &&other)
{
    Display = other.Display;
    MaxIter = other.MaxIter;
    FunctionMinimizerType = other.FunctionMinimizerType;
    Step = other.Step;
    Tolerance = other.Tolerance;
}

template <typename T>
optimizationParameters<T> &optimizationParameters<T>::operator=(optimizationParameters<T> const &other)
{
    Display = other.Display;
    MaxIter = other.MaxIter;
    FunctionMinimizerType = other.FunctionMinimizerType;
    Step = other.Step;
    Tolerance = other.Tolerance;
    return *this;
}

template <typename T>
optimizationParameters<T> &optimizationParameters<T>::operator=(optimizationParameters<T> &&other)
{
    Display = other.Display;
    MaxIter = other.MaxIter;
    FunctionMinimizerType = other.FunctionMinimizerType;
    Step = other.Step;
    Tolerance = other.Tolerance;
    return *this;
}

template <typename T>
optimizationParameters<T>::~optimizationParameters() {}

template <typename T>
void optimizationParameters<T>::reset()
{
    MaxIter = 100;
    Display = 0;
    FunctionMinimizerType = 2;
    Tolerance = 1e-6;
    Step = 1e-5;
}

template <typename T>
void optimizationParameters<T>::reset(int const maxIter, int const display, int const functionMinimizerType, T const tolerance, T const step)
{
    MaxIter = maxIter;
    Display = display;
    FunctionMinimizerType = functionMinimizerType;
    Tolerance = tolerance;
    Step = step;
}

} // namespace tmcmc
} // namespace umuq

#endif // UMUQ_STDATA
