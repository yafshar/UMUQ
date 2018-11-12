#ifndef UMUQ_PRIORDISTRIBUTION_H
#define UMUQ_PRIORDISTRIBUTION_H

#include "core/core.hpp"
#include "numerics/density.hpp"
#include "numerics/random/psrandom.hpp"

namespace umuq
{

/*! \enum priorTypes
 * \ingroup Inference_Module
 * 
 * \brief Prior distribution types currently supported in %UMUQ
 * 
 */
enum priorTypes
{
    /*! \link umuq::density::uniformDistribution UNIFORM \endlink */
    UNIFORM = 0,
    /*! \link umuq::density::gaussianDistribution GAUSSIAN \endlink */
    GAUSSIAN = 1,
    /*! \link umuq::density::exponentialDistribution EXPONENTIAL \endlink */
    EXPONENTIAL = 2,
    /*! \link umuq::density::gammaDistribution GAMMA \endlink */
    GAMMA = 3,
    /*! COMPOSITE  */
    COMPOSITE = 4
};

/*! \class priorDistribution
 * \ingroup Inference_Module
 * 
 * \brief Prior distribution which is one of the:
 * 
 * <table>
 * <caption id="multi_row">Prior distribution types</caption>
 * <tr><th> Index number <th> Prior distribution type        
 * <tr><td> 0       <td> UNIFORM  
 * <tr><td> 1       <td> GAUSSIAN    
 * <tr><td> 2       <td> EXPONENTIAL     
 * <tr><td> 3       <td> GAMMA 
 * <tr><td> 4       <td> COMPOSITE
 * </table>
 * 
 * \tparam T Data type 
 * 
 * USE: <br>
 * To use the priorDistribution object:
 * - First, construct a new prior Distribution object with problem dimension and the prior type. <br>
 *   In case, the prior type is not known yet, you should reset the priorDistribution later in the code with the 
 *   correct problem dimension and corresponding prior type. <br>
 *   \sa priorTypes. <br>
 *   \sa reset. <br>
 * 
 * - Second, set the priorDistribution parameters. <br> 
 *   \sa set.
 * 
 * - Third, call the member function. <br>
 *   You can call the probability density function (pdf). <br>
 *    or <br>
 *   logarithm probability density function (logpdf). <br>
 *   \sa pdf. <br>
 *   \sa logpdf. <br>
 * 
 *   IF you also need samples from the desired distribution, you must set the Random Number Generator object. <br>
 *   otherwise <br> 
 *   you can not use sample member function (sample). <br>
 *   \sa setRandomGenerator.<br>   
 *   \sa sample. <br>
 * 
 * - Forth, call any other member function.
 */
template <typename T>
class priorDistribution
{
  public:
    /*!
     * \brief Construct a new prior Distribution object
     * 
     */
    priorDistribution();

    /*!
     * \brief Construct a new prior Distribution object
     * 
     * \param probdim  Problem dimension
     * \param prior    Prior type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite)
     */
    priorDistribution(int const probdim, int const prior = 0);

    /*!
     * \brief Move constructor, construct a new priorDistribution object from input priorDistribution object
     * 
     * \param other priorDistribution object
     */
    priorDistribution(priorDistribution<T> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other priorDistribution object
     * 
     * \returns priorDistribution<T>& 
     */
    priorDistribution<T> &operator=(priorDistribution<T> &&other);

    /*!
     * \brief Destroy the prior Distribution object
     * 
     */
    ~priorDistribution();

    /*!
     * \brief Reset the priorDistribution object size & type
     * 
     * \param probdim  Problem dimension
     * \param prior    Prior type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite)
     * 
     * \returns true 
     * \returns false  If there is not enough memory or wrong prior type
     */
    bool reset(int const probdim, int const prior = 0);

    /*!
     * \brief Set the priorDistribution parameters
     * 
     * \param Param1          First parameter for a prior distribution  
     * \param Param2          Second parameter for a prior distribution  
     * \param compositeprior  Composite priors type
     * 
     * \returns true 
     * \returns false If it encounters an unexpected problem
     */
    bool set(T const *Param1, T const *Param2, int const *compositeprior = nullptr);

    /*!
     * \brief Set the priorDistribution parameters
     * 
     * \param Param1          First parameter for a prior distribution  
     * \param Param2          Second parameter for a prior distribution  
     * \param compositeprior  Composite priors type
     * 
     * \returns true 
     * \returns false If it encounters an unexpected problem
     */
    bool set(std::vector<T> const &Param1, std::vector<T> const &Param2, std::vector<int> const &compositeprior = EmptyVector<int>);

    /*!
     * \brief Set the Random Number Generator object to 
     * 
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     * 
     * \returns true 
     * \returns false If it encounters an unexpected problem
     */
    bool setRandomGenerator(psrandom<T> *PRNG);

    /*!
     * \brief Get the dimension
     * 
     * \returns int Dimension of the problem
     */
    inline int getDim();

    /*!
     * \brief Get the prior type
     * 
     * \returns int prior type
     */
    inline int getpriorType();

    /*!
     * \brief Get the Prior Types for the composite prior
     * 
     * \returns int* Prior Types
     */
    inline int *getPriorTypes();

    /*!
     * \brief Probability density function (pdf)
     * 
     * \param x  Input point
     *  
     * \returns T Returns the probability density function (pdf) evaluated in x
     */
    T pdf(T const *x);

    /*!
     * \brief Probability density function (pdf)
     * 
     * \param x  Input point
     *  
     * \returns T Returns the probability density function (pdf) evaluated in x
     */
    T pdf(std::vector<T> const &x);

    /*!
     * \brief Logarithm of the probability density function
     * 
     * \param x  Input point
     * 
     * \returns T Returns the logarithm probability density function (pdf) evaluated in x
     */
    T logpdf(T const *x);

    /*!
     * \brief Logarithm of the probability density function
     * 
     * \param x  Input point
     * 
     * \returns T Returns the logarithm probability density function (pdf) evaluated in x
     */
    T logpdf(std::vector<T> const &x);

    /*!
     * \brief Create samples based on the prior distribution type
     * 
     * \param x Samples 
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool sample(T *x);

    /*!
     * \brief Create samples based on the prior distribution type
     * 
     * \param x Samples 
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool sample(std::vector<T> &x);

  protected:
    /*!
     * \brief Delete a priorDistribution object copy construction
     * 
     * Make it noncopyable.
     */
    priorDistribution(priorDistribution<T> const &) = delete;

    /*!
     * \brief Delete a priorDistribution object assignment
     * 
     * Make it nonassignable
     * 
     * \returns priorDistribution<T>& 
     */
    priorDistribution<T> &operator=(priorDistribution<T> const &) = delete;

  private:
    //! Problem Dimension
    int nDim;

    /*!
     * Prior type which is one of : <br>
     * 0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite
     */
    int priorType;

    //! Composite distribution prior
    std::unique_ptr<int[]> compositePrior;

  private:
    //! Flat (Uniform) distribution
    std::unique_ptr<uniformDistribution<T>> uniformDist;

    //! The Multivariate Gaussian Distribution
    std::unique_ptr<multivariategaussianDistribution<T>> multivariategaussianDist;

    //! The exponential distribution
    std::unique_ptr<exponentialDistribution<T>> exponentialDist;

    //! The Gamma distribution
    std::unique_ptr<gammaDistribution<T>> gammaDist;

    //! The Gaussian distribution
    std::unique_ptr<gaussianDistribution<T>> gaussianDist;

  private:
    /*!
     * The below data are only used for composite prior distribution
     * indexing of the points
     */
    std::vector<int> uniformIndex;
    std::vector<int> gaussianIndex;
    std::vector<int> exponentialIndex;
    std::vector<int> gammaIndex;

    //! Input points
    std::vector<T> uniformPoints;
    std::vector<T> gaussianPoints;
    std::vector<T> exponentialPoints;
    std::vector<T> gammaPoints;
};

template <typename T>
priorDistribution<T>::priorDistribution() : nDim(0),
                                            priorType(priorTypes::UNIFORM),
                                            compositePrior(nullptr),
                                            uniformDist(nullptr),
                                            multivariategaussianDist(nullptr),
                                            exponentialDist(nullptr),
                                            gammaDist(nullptr),
                                            gaussianDist(nullptr) {}

template <typename T>
priorDistribution<T>::priorDistribution(int const probdim, int const prior) : nDim(probdim),
                                                                              priorType(prior),
                                                                              compositePrior(nullptr),
                                                                              uniformDist(nullptr),
                                                                              multivariategaussianDist(nullptr),
                                                                              exponentialDist(nullptr),
                                                                              gammaDist(nullptr),
                                                                              gaussianDist(nullptr)
{
    switch (priorType)
    {
    case priorTypes::UNIFORM:
        break;
    case priorTypes::GAUSSIAN:
        break;
    case priorTypes::EXPONENTIAL:
        break;
    case priorTypes::GAMMA:
        break;
    case priorTypes::COMPOSITE:
        break;
    default:
        UMUQFAIL("Unknown prior distribution type!");
        break;
    };
}

template <typename T>
priorDistribution<T>::priorDistribution(priorDistribution<T> &&other) : nDim(other.nDim),
                                                                        priorType(other.priorType),
                                                                        compositePrior(std::move(other.compositePrior)),
                                                                        uniformDist(std::move(other.uniformDist)),
                                                                        multivariategaussianDist(std::move(other.multivariategaussianDist)),
                                                                        exponentialDist(std::move(other.exponentialDist)),
                                                                        gammaDist(std::move(other.gammaDist)),
                                                                        gaussianDist(std::move(other.gaussianDist)),
                                                                        uniformIndex(std::move(other.uniformIndex)),
                                                                        gaussianIndex(std::move(other.gaussianIndex)),
                                                                        exponentialIndex(std::move(other.exponentialIndex)),
                                                                        gammaIndex(std::move(other.gammaIndex)),
                                                                        uniformPoints(std::move(other.uniformPoints)),
                                                                        gaussianPoints(std::move(other.gaussianPoints)),
                                                                        exponentialPoints(std::move(other.exponentialPoints)),
                                                                        gammaPoints(std::move(other.gammaPoints))
{
    switch (priorType)
    {
    case priorTypes::UNIFORM:
        break;
    case priorTypes::GAUSSIAN:
        break;
    case priorTypes::EXPONENTIAL:
        break;
    case priorTypes::GAMMA:
        break;
    case priorTypes::COMPOSITE:
        break;
    default:
        UMUQFAIL("Unknown prior distribution type!");
        break;
    };
}

template <typename T>
priorDistribution<T> &priorDistribution<T>::operator=(priorDistribution<T> &&other)
{
    nDim = other.nDim;
    priorType = other.priorType;
    switch (priorType)
    {
    case priorTypes::UNIFORM:
        break;
    case priorTypes::GAUSSIAN:
        break;
    case priorTypes::EXPONENTIAL:
        break;
    case priorTypes::GAMMA:
        break;
    case priorTypes::COMPOSITE:
        break;
    default:
        UMUQFAIL("Unknown prior distribution type!");
        break;
    };
    compositePrior = std::move(other.compositePrior);
    uniformDist = std::move(other.uniformDist);
    multivariategaussianDist = std::move(other.multivariategaussianDist);
    exponentialDist = std::move(other.exponentialDist);
    gammaDist = std::move(other.gammaDist);
    gaussianDist = std::move(other.gaussianDist);
    uniformIndex = std::move(other.uniformIndex);
    gaussianIndex = std::move(other.gaussianIndex);
    exponentialIndex = std::move(other.exponentialIndex);
    gammaIndex = std::move(other.gammaIndex);
    uniformPoints = std::move(other.uniformPoints);
    gaussianPoints = std::move(other.gaussianPoints);
    exponentialPoints = std::move(other.exponentialPoints);
    gammaPoints = std::move(other.gammaPoints);

    return *this;
}

template <typename T>
priorDistribution<T>::~priorDistribution() {}

template <typename T>
bool priorDistribution<T>::reset(int const probdim, int const prior)
{
    nDim = probdim;
    priorType = prior;
    switch (priorType)
    {
    case priorTypes::UNIFORM:
        break;
    case priorTypes::GAUSSIAN:
        break;
    case priorTypes::EXPONENTIAL:
        break;
    case priorTypes::GAMMA:
        break;
    case priorTypes::COMPOSITE:
        break;
    default:
        UMUQFAILRETURN("Unknown prior distribution type!");
        break;
    };
    if (compositePrior)
    {
        compositePrior.reset(nullptr);
    }
    if (uniformDist)
    {
        uniformDist.reset(nullptr);
    }
    if (multivariategaussianDist)
    {
        multivariategaussianDist.reset(nullptr);
    }
    if (exponentialDist)
    {
        exponentialDist.reset(nullptr);
    }
    if (gammaDist)
    {
        gammaDist.reset(nullptr);
    }
    if (gaussianDist)
    {
        gaussianDist.reset(nullptr);
    }
    return true;
}

template <typename T>
bool priorDistribution<T>::set(T const *Param1, T const *Param2, int const *compositeprior)
{
    switch (priorType)
    {
    case priorTypes::UNIFORM:
        try
        {
            uniformDist.reset(new uniformDistribution<T>(Param1, Param2, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case priorTypes::GAUSSIAN:
        try
        {
            multivariategaussianDist.reset(new multivariategaussianDistribution<T>(Param1, Param2, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case priorTypes::EXPONENTIAL:
        try
        {
            exponentialDist.reset(new exponentialDistribution<T>(Param1, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case priorTypes::GAMMA:
        try
        {
            gammaDist.reset(new gammaDistribution<T>(Param1, Param2, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case priorTypes::COMPOSITE:
    {
        if (compositeprior)
        {
            try
            {
                compositePrior.reset(new int[nDim]());
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }

            int *p = const_cast<int *>(compositeprior);
            std::copy(p, p + nDim, compositePrior.get());
        }
        else
        {
            UMUQFAILRETURN("Failed to provide composite prior types for each dimension!");
        }

        if (multivariategaussianDist)
        {
            multivariategaussianDist.reset(nullptr);
        }

        int nUNIFORM(0);
        int nGAUSSIAN(0);
        int nEXPONENTIAL(0);
        int nGAMMA(0);

        for (int i = 0; i < nDim; i++)
        {
            switch (compositePrior[i])
            {
            case priorTypes::UNIFORM:
                nUNIFORM++;
                break;
            case priorTypes::GAUSSIAN:
                nGAUSSIAN++;
                break;
            case priorTypes::EXPONENTIAL:
                nEXPONENTIAL++;
                break;
            case priorTypes::GAMMA:
                nGAMMA++;
                break;
            default:
                UMUQFAILRETURN("Unknown prior distribution type!");
                break;
            };
        }

        if (nUNIFORM)
        {
            uniformIndex.resize(nUNIFORM);
            uniformPoints.resize(nUNIFORM);
        }
        if (nGAUSSIAN)
        {
            gaussianIndex.resize(nGAUSSIAN);
            gaussianPoints.resize(nGAUSSIAN);
        }
        if (nEXPONENTIAL)
        {
            exponentialIndex.resize(nEXPONENTIAL);
            exponentialPoints.resize(nEXPONENTIAL);
        }
        if (nGAMMA)
        {
            gammaIndex.resize(nGAMMA);
            gammaPoints.resize(nGAMMA);
        }

        std::vector<T> uparam1(nUNIFORM);
        std::vector<T> uparam2(nUNIFORM);
        std::vector<T> nparam1(nGAUSSIAN);
        std::vector<T> nparam2(nGAUSSIAN);
        std::vector<T> eparam1(nEXPONENTIAL);
        std::vector<T> gparam1(nGAMMA);
        std::vector<T> gparam2(nGAMMA);

        nUNIFORM = 0;
        nGAUSSIAN = 0;
        nEXPONENTIAL = 0;
        nGAMMA = 0;

        for (int i = 0; i < nDim; i++)
        {
            switch (compositePrior[i])
            {
            case priorTypes::UNIFORM:
                uparam1[nUNIFORM] = Param1[i];
                uparam2[nUNIFORM] = Param2[i];
                uniformIndex[nUNIFORM] = i;
                nUNIFORM++;
                break;
            case priorTypes::GAUSSIAN:
                nparam1[nGAUSSIAN] = Param1[i];
                nparam2[nGAUSSIAN] = Param2[i];
                gaussianIndex[nGAUSSIAN] = i;
                nGAUSSIAN++;
                break;
            case priorTypes::EXPONENTIAL:
                eparam1[nEXPONENTIAL] = Param1[i];
                exponentialIndex[nEXPONENTIAL] = i;
                nEXPONENTIAL++;
                break;
            case priorTypes::GAMMA:
                gparam1[nGAMMA] = Param1[i];
                gparam2[nGAMMA] = Param2[i];
                gammaIndex[nGAMMA] = i;
                nGAMMA++;
                break;
            };
        }

        if (nUNIFORM)
        {
            try
            {
                uniformDist.reset(new uniformDistribution<T>(uparam1.data(), uparam2.data(), nUNIFORM * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (uniformDist)
            {
                uniformDist.reset(nullptr);
            }
        }
        if (nGAUSSIAN)
        {
            try
            {
                gaussianDist.reset(new gaussianDistribution<T>(nparam1.data(), nparam2.data(), nGAUSSIAN * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            gaussianDist.reset(nullptr);
        }
        if (nEXPONENTIAL)
        {
            try
            {
                exponentialDist.reset(new exponentialDistribution<T>(eparam1.data(), nEXPONENTIAL));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (exponentialDist)
            {
                exponentialDist.reset(nullptr);
            }
        }
        if (nGAMMA)
        {
            try
            {
                gammaDist.reset(new gammaDistribution<T>(gparam1.data(), gparam2.data(), nGAMMA * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (gammaDist)
            {
                gammaDist.reset(nullptr);
            }
        }
    }

    break;
    default:
        UMUQFAILRETURN("Unknown prior distribution type!");
        break;
    };

    return true;
}

template <typename T>
bool priorDistribution<T>::set(std::vector<T> const &Param1, std::vector<T> const &Param2, std::vector<int> const &compositeprior)
{
    return compositeprior.size() ? set(Param1.data(), Param2.data(), compositeprior.data()) : set(Param1.data(), Param2.data());
}

template <typename T>
bool priorDistribution<T>::setRandomGenerator(psrandom<T> *PRNG)
{
    if (PRNG)
    {
        if (PRNG_initialized)
        {
            switch (priorType)
            {
            case priorTypes::UNIFORM:
                if (uniformDist)
                {
                    return uniformDist->setRandomGenerator(PRNG);
                }
                break;
            case priorTypes::GAUSSIAN:
                if (multivariategaussianDist)
                {
                    return multivariategaussianDist->setRandomGenerator(PRNG);
                }
                break;
            case priorTypes::EXPONENTIAL:
                if (exponentialDist)
                {
                    return exponentialDist->setRandomGenerator(PRNG);
                }
                break;
            case priorTypes::GAMMA:
                if (gammaDist)
                {
                    return gammaDist->setRandomGenerator(PRNG);
                }
                break;
            case priorTypes::COMPOSITE:
            {
                bool cstatus = true;
                bool unfmstatus = false;
                bool gausstatus = false;
                bool expostatus = false;
                bool gammstatus = false;
                for (int i = 0; i < nDim; i++)
                {
                    switch (compositePrior[i])
                    {
                    case priorTypes::UNIFORM:
                        if (!unfmstatus)
                        {
                            if (uniformDist)
                            {
                                cstatus = cstatus && uniformDist->setRandomGenerator(PRNG);
                                unfmstatus = cstatus;
                            }
                            else
                            {
                                UMUQFAILRETURN("PriorDistribution parameters are not set!");
                            }
                        }
                        break;
                    case priorTypes::GAUSSIAN:
                        if (!gausstatus)
                        {
                            if (gaussianDist)
                            {
                                cstatus = cstatus && gaussianDist->setRandomGenerator(PRNG);
                                gausstatus = cstatus;
                            }
                            else
                            {
                                UMUQFAILRETURN("PriorDistribution parameters are not set!");
                            }
                        }
                        break;
                    case priorTypes::EXPONENTIAL:
                        if (!expostatus)
                        {
                            if (exponentialDist)
                            {
                                cstatus = cstatus && exponentialDist->setRandomGenerator(PRNG);
                                expostatus = cstatus;
                            }
                            else
                            {
                                UMUQFAILRETURN("PriorDistribution parameters are not set!");
                            }
                        }
                        break;
                    case priorTypes::GAMMA:
                        if (!gammstatus)
                        {
                            if (gammaDist)
                            {
                                cstatus = cstatus && gammaDist->setRandomGenerator(PRNG);
                                gammstatus = cstatus;
                            }
                            else
                            {
                                UMUQFAILRETURN("PriorDistribution parameters are not set!");
                            }
                        }
                        break;
                    };
                }
                return cstatus;
            }
            break;
            default:
                UMUQFAILRETURN("Unknown prior distribution type!");
                break;
            };
            UMUQFAILRETURN("Prior Distribution is not set!");
        }
        UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to any prior distribution!");
    }
    UMUQFAILRETURN("The pseudo-random number generator is not assigned!");
}

template <typename T>
inline int priorDistribution<T>::getDim()
{
    return nDim;
}

template <typename T>
inline int priorDistribution<T>::getpriorType()
{
    return priorType;
}

template <typename T>
inline int *priorDistribution<T>::getPriorTypes()
{
    return compositePrior.get();
}

template <typename T>
T priorDistribution<T>::pdf(T const *x)
{
    switch (priorType)
    {
    case priorTypes::UNIFORM:
        return uniformDist->f(x);
        break;
    case priorTypes::GAUSSIAN:
        return multivariategaussianDist->f(x);
        break;
    case priorTypes::EXPONENTIAL:
        return exponentialDist->f(x);
        break;
    case priorTypes::GAMMA:
        return gammaDist->f(x);
        break;
    case priorTypes::COMPOSITE:
        T sum(1);
        if (uniformDist)
        {
            int j(0);
            for (auto i : uniformIndex)
            {
                uniformPoints[j++] = x[i];
            }
            sum *= uniformDist->f(uniformPoints.data());
        }
        if (gaussianDist)
        {
            int j(0);
            for (auto i : gaussianIndex)
            {
                gaussianPoints[j++] = x[i];
            }
            sum *= gaussianDist->f(gaussianPoints.data());
        }
        if (exponentialDist)
        {
            int j(0);
            for (auto i : exponentialIndex)
            {
                exponentialPoints[j++] = x[i];
            }
            sum *= exponentialDist->f(exponentialPoints.data());
        }
        if (gammaDist)
        {
            int j(0);
            for (auto i : gammaIndex)
            {
                gammaPoints[j++] = x[i];
            }
            sum *= gammaDist->f(gammaPoints.data());
        }
        return sum;
        break;
    }
    UMUQFAIL("Unknown Prior type!");
}

template <typename T>
T priorDistribution<T>::pdf(std::vector<T> const &x)
{
    return pdf(x.data());
}

template <typename T>
T priorDistribution<T>::logpdf(T const *x)
{
    switch (priorType)
    {
    case priorTypes::UNIFORM:
        return uniformDist->lf(x);
        break;
    case priorTypes::GAUSSIAN:
        return multivariategaussianDist->lf(x);
        break;
    case priorTypes::EXPONENTIAL:
        return exponentialDist->lf(x);
        break;
    case priorTypes::GAMMA:
        return gammaDist->lf(x);
        break;
    case priorTypes::COMPOSITE:
        T sum(0);
        if (uniformDist)
        {
            int j(0);
            for (auto i : uniformIndex)
            {
                uniformPoints[j++] = x[i];
            }
            sum += uniformDist->lf(uniformPoints.data());
        }
        if (gaussianDist)
        {
            int j(0);
            for (auto i : gaussianIndex)
            {
                gaussianPoints[j++] = x[i];
            }
            sum += gaussianDist->lf(gaussianPoints.data());
        }
        if (exponentialDist)
        {
            int j(0);
            for (auto i : exponentialIndex)
            {
                exponentialPoints[j++] = x[i];
            }
            sum += exponentialDist->lf(exponentialPoints.data());
        }
        if (gammaDist)
        {
            int j(0);
            for (auto i : gammaIndex)
            {
                gammaPoints[j++] = x[i];
            }
            sum += gammaDist->lf(gammaPoints.data());
        }
        return sum;
        break;
    }
    UMUQFAIL("Unknown Prior type!");
}

template <typename T>
T priorDistribution<T>::logpdf(std::vector<T> const &x)
{
    return logpdf(x.data());
}

template <typename T>
bool priorDistribution<T>::sample(T *x)
{
    switch (priorType)
    {
    case priorTypes::UNIFORM:
        return uniformDist->sample(x);
        break;
    case priorTypes::GAUSSIAN:
        return multivariategaussianDist->sample(x);
        break;
    case priorTypes::EXPONENTIAL:
        return exponentialDist->sample(x);
        break;
    case priorTypes::GAMMA:
        return gammaDist->sample(x);
        break;
    case priorTypes::COMPOSITE:
        if (uniformDist)
        {
            if (uniformDist->sample(uniformPoints.data()))
            {
                int j(0);
                for (auto i : uniformIndex)
                {
                    x[i] = uniformPoints[j++];
                }
            }
        }
        if (gaussianDist)
        {
            if (gaussianDist->sample(gaussianPoints.data()))
            {
                int j(0);
                for (auto i : gaussianIndex)
                {
                    x[i] = gaussianPoints[j++];
                }
            }
        }
        if (exponentialDist)
        {
            if (exponentialDist->sample(exponentialPoints.data()))
            {
                int j(0);
                for (auto i : exponentialIndex)
                {
                    x[i] = exponentialPoints[j++];
                }
            }
        }
        if (gammaDist)
        {
            if (gammaDist->sample(gammaPoints.data()))
            {
                int j(0);
                for (auto i : gammaIndex)
                {
                    x[i] = gammaPoints[j++];
                }
            }
        }
        return true;
        break;
    }
    UMUQFAIL("Unknown Prior type!");
}

template <typename T>
bool priorDistribution<T>::sample(std::vector<T> &x)
{
    return sample(x.data());
}

} // namespace umuq

#endif // UMUQ_PRIORDISTRIBUTION
