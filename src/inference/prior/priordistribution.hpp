#ifndef UMUQ_PRIORDISTRIBUTION_H
#define UMUQ_PRIORDISTRIBUTION_H

#include "core/core.hpp"
#include "numerics/density.hpp"
#include "numerics/random/psrandom.hpp"

namespace umuq
{

/*!
 * \brief Prior distribution types
 * 
 * Currenly we have these types:
 * \b UNIFORM \sa uniformDistribution
 * \b GAUSSIAN \sa gaussianDistribution
 * \b EXPONENTIAL \sa exponentialDistribution
 * \b GAMMA \sa gammaDistribution
 * \b COMPOSITE
 */
enum priorTypes
{
    UNIFORM = 0,
    GAUSSIAN = 1,
    EXPONENTIAL = 2,
    GAMMA = 3,
    COMPOSITE = 4
};

/*!
 * \brief Prior distribution which is one of the:
 * 0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite
 * 
 * \tparam T Data type 
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
     * \param other  Input priorDistribution object
     */
    priorDistribution(priorDistribution<T> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other priorDistribution object
     * 
     * \return priorDistribution<T>& 
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
     * \return true 
     * \return false  If there is not enough memory or wrong prior type
     */
    bool reset(int const probdim, int const prior = 0);

    /*!
     * \brief Set the priorDistribution parameters
     * 
     * \param Param1          First parameter for a prior distribution  
     * \param Param2          Second parameter for a prior distribution  
     * \param compositeprior  Composite priors type
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    bool set(T const *Param1, T const *Param2, int const *compositeprior = nullptr);
    bool set(std::vector<T> const &Param1, std::vector<T> const &Param2, std::vector<int> const &compositeprior = std::vector<int>{});

    /*!
     * \brief Set the Random Number Generator object to 
     * 
     * \param PRNG  Pseudo-random number object \sa psrandom
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<T> *PRNG);

    /*!
     * \brief Get the dimension
     * 
     * \return int Dimension of the pronlem
     */
    inline int getDim();

    /*!
     * \brief Get the prior type
     * 
     * \return int prior type
     */
    inline int getpriorType();

    /*!
     * \brief Get the Prior Types for the composite prior
     * 
     * \return int* Prior Types
     */
    inline int *getPriorTypes();

    /*!
     * \brief Probability density function (pdf)
     * 
     * \param x  Input point
     *  
     * \return T Returns the probability density function (pdf) evaluated in x
     */
    T pdf(T const *x);
    T pdf(std::vector<T> const &x);

    /*!
     * \brief Logarithm of the probability density function
     * 
     * \param x  Input point
     * 
     * \return T Returns the logarithm probability density function (pdf) evaluated in x
     */
    T logpdf(T const *x);
    T logpdf(std::vector<T> const &x);

    /*!
     * \brief Create samples based on the prior distribution type
     * 
     * \param x Samples 
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    bool sample(T *x);
    bool sample(std::vector<T> &x);

  private:
    // Make it noncopyable
    priorDistribution(priorDistribution<T> const &) = delete;

    // Make it not assignable
    priorDistribution<T> &operator=(priorDistribution<T> const &) = delete;

  private:
    //! Problem Dimension
    int nDim;

    //! Prior type which is :
    //! 0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite
    int priorType;

    //! Composite distribution as a prior
    std::unique_ptr<int[]> compositePrior;

  private:
    //! Flat (Uniform) distribution
    std::unique_ptr<uniformDistribution<T>> unfm;

    //! The Multivariate Gaussian Distribution
    std::unique_ptr<multivariategaussianDistribution<T>> mvnp;

    //! The exponential distribution
    std::unique_ptr<exponentialDistribution<T>> expo;

    //! The Gamma distribution
    std::unique_ptr<gammaDistribution<T>> gamm;

  private:
    //! The Gaussian distribution
    std::unique_ptr<gaussianDistribution<T>> gaus;

  private:
    //! Pointer to psudo random number generator object
    psrandom<T> *prng;

  private:
    //! The below data are only used for composite prior distribution
    //! Index of the points
    std::vector<int> unfmIndex;
    std::vector<int> gausIndex;
    std::vector<int> expoIndex;
    std::vector<int> gammIndex;

    //! Input points
    std::vector<T> unfmX;
    std::vector<T> gausX;
    std::vector<T> expoX;
    std::vector<T> gammX;
};

template <typename T>
priorDistribution<T>::priorDistribution() : nDim(0),
                                            priorType(priorTypes::UNIFORM),
                                            unfm(nullptr),
                                            mvnp(nullptr),
                                            expo(nullptr),
                                            gamm(nullptr),
                                            gaus(nullptr),
                                            prng(nullptr) {}

template <typename T>
priorDistribution<T>::priorDistribution(int const probdim, int const prior) : nDim(probdim),
                                                                              priorType(prior),
                                                                              unfm(nullptr),
                                                                              mvnp(nullptr),
                                                                              expo(nullptr),
                                                                              gamm(nullptr),
                                                                              gaus(nullptr),
                                                                              prng(nullptr)
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
                                                                        unfm(std::move(other.unfm)),
                                                                        mvnp(std::move(other.mvnp)),
                                                                        expo(std::move(other.expo)),
                                                                        gamm(std::move(other.gamm)),
                                                                        gaus(std::move(other.gaus)),
                                                                        prng(other.prng),
                                                                        unfmIndex(std::move(other.unfmIndex)),
                                                                        gausIndex(std::move(other.gausIndex)),
                                                                        expoIndex(std::move(other.expoIndex)),
                                                                        gammIndex(std::move(other.gammIndex)),
                                                                        unfmX(std::move(other.unfmX)),
                                                                        gausX(std::move(other.gausX)),
                                                                        expoX(std::move(other.expoX)),
                                                                        gammX(std::move(other.gammX))
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
    unfm = std::move(other.unfm);
    mvnp = std::move(other.mvnp);
    expo = std::move(other.expo);
    gamm = std::move(other.gamm);
    gaus = std::move(other.gaus);
    prng = other.prng;
    unfmIndex = std::move(other.unfmIndex);
    gausIndex = std::move(other.gausIndex);
    expoIndex = std::move(other.expoIndex);
    gammIndex = std::move(other.gammIndex);
    unfmX = std::move(other.unfmX);
    gausX = std::move(other.gausX);
    expoX = std::move(other.expoX);
    gammX = std::move(other.gammX);

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
            unfm.reset(new uniformDistribution<T>(Param1, Param2, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case priorTypes::GAUSSIAN:
        try
        {
            mvnp.reset(new multivariategaussianDistribution<T>(Param1, Param2, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case priorTypes::EXPONENTIAL:
        try
        {
            expo.reset(new exponentialDistribution<T>(Param1, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case priorTypes::GAMMA:
        try
        {
            gamm.reset(new gammaDistribution<T>(Param1, Param2, nDim * 2));
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

        if (mvnp)
        {
            mvnp.reset(nullptr);
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

        unfmIndex.resize(nUNIFORM);
        gausIndex.resize(nGAUSSIAN);
        expoIndex.resize(nEXPONENTIAL);
        gammIndex.resize(nGAMMA);

        unfmX.resize(nUNIFORM);
        gausX.resize(nGAUSSIAN);
        expoX.resize(nEXPONENTIAL);
        gammX.resize(nGAMMA);

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
                unfmIndex[nUNIFORM] = i;
                nUNIFORM++;
                break;
            case priorTypes::GAUSSIAN:
                nparam1[nGAUSSIAN] = Param1[i];
                nparam1[nGAUSSIAN] = Param2[i];
                gausIndex[nGAUSSIAN] = i;
                nGAUSSIAN++;
                break;
            case priorTypes::EXPONENTIAL:
                eparam1[nEXPONENTIAL] = Param1[i];
                expoIndex[nEXPONENTIAL] = i;
                nEXPONENTIAL++;
                break;
            case priorTypes::GAMMA:
                gparam1[nGAMMA] = Param1[i];
                gparam1[nGAMMA] = Param2[i];
                gammIndex[nGAMMA] = i;
                nGAMMA++;
                break;
            };
        }

        if (nUNIFORM)
        {
            try
            {
                unfm.reset(new uniformDistribution<T>(uparam1.data(), uparam2.data(), nUNIFORM * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (unfm)
            {
                unfm.reset(nullptr);
            }
        }
        if (nGAUSSIAN)
        {
            try
            {
                gaus.reset(new gaussianDistribution<T>(nparam1.data(), nparam2.data(), nGAUSSIAN * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            gaus.reset(nullptr);
        }
        if (nEXPONENTIAL)
        {
            try
            {
                expo.reset(new exponentialDistribution<T>(eparam1.data(), nEXPONENTIAL));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (expo)
            {
                expo.reset(nullptr);
            }
        }
        if (nGAMMA)
        {
            try
            {
                gamm.reset(new gammaDistribution<T>(gparam1.data(), gparam2.data(), nGAMMA * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (gamm)
            {
                gamm.reset(nullptr);
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
    return set(Param1.data(), Param2.data(), compositeprior.data());
}

template <typename T>
inline bool priorDistribution<T>::setRandomGenerator(psrandom<T> *PRNG)
{
    if (PRNG)
    {
        prng = PRNG;
        return true;
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!")
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
        return unfm->f(x);
        break;
    case priorTypes::GAUSSIAN:
        return mvnp->f(x);
        break;
    case priorTypes::EXPONENTIAL:
        return expo->f(x);
        break;
    case priorTypes::GAMMA:
        return gamm->f(x);
        break;
    case priorTypes::COMPOSITE:
        T sum(1);
        if (unfm)
        {
            int j(0);
            for (auto i : unfmIndex)
            {
                unfmX[j++] = x[i];
            }
            sum *= unfm->f(unfmX.data());
        }
        if (gaus)
        {
            int j(0);
            for (auto i : gausIndex)
            {
                gausX[j++] = x[i];
            }
            sum *= gaus->f(gausX.data());
        }
        if (expo)
        {
            int j(0);
            for (auto i : expoIndex)
            {
                expoX[j++] = x[i];
            }
            sum *= expo->f(expoX.data());
        }
        if (gamm)
        {
            int j(0);
            for (auto i : gammIndex)
            {
                gammX[j++] = x[i];
            }
            sum *= gamm->f(gammX.data());
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
        return unfm->lf(x);
        break;
    case priorTypes::GAUSSIAN:
        return mvnp->lf(x);
        break;
    case priorTypes::EXPONENTIAL:
        return expo->lf(x);
        break;
    case priorTypes::GAMMA:
        return gamm->lf(x);
        break;
    case priorTypes::COMPOSITE:
        T sum(0);
        if (unfm)
        {
            int j(0);
            for (auto i : unfmIndex)
            {
                unfmX[j++] = x[i];
            }
            sum += unfm->lf(unfmX.data());
        }
        if (gaus)
        {
            int j(0);
            for (auto i : gausIndex)
            {
                gausX[j++] = x[i];
            }
            sum += gaus->lf(gausX.data());
        }
        if (expo)
        {
            int j(0);
            for (auto i : expoIndex)
            {
                expoX[j++] = x[i];
            }
            sum += expo->lf(expoX.data());
        }
        if (gamm)
        {
            int j(0);
            for (auto i : gammIndex)
            {
                gammX[j++] = x[i];
            }
            sum += gamm->lf(gammX.data());
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
    if (prng)
    {
        switch (priorType)
        {
        case priorTypes::UNIFORM:
            for (int i=0;i<nDim;i++)
            {
                x[i] = prng->unirnd();
            }
            return unfm->lf(x);
            break;
        case priorTypes::GAUSSIAN:
            return mvnp->lf(x);
            break;
        case priorTypes::EXPONENTIAL:
            return expo->lf(x);
            break;
        case priorTypes::GAMMA:
            return gamm->lf(x);
            break;
        case priorTypes::COMPOSITE:
            T sum(0);
            if (unfm)
            {
                int j(0);
                for (auto i : unfmIndex)
                {
                    unfmX[j++] = x[i];
                }
                sum += unfm->lf(unfmX.data());
            }
            if (gaus)
            {
                int j(0);
                for (auto i : gausIndex)
                {
                    gausX[j++] = x[i];
                }
                sum += gaus->lf(gausX.data());
            }
            if (expo)
            {
                int j(0);
                for (auto i : expoIndex)
                {
                    expoX[j++] = x[i];
                }
                sum += expo->lf(expoX.data());
            }
            if (gamm)
            {
                int j(0);
                for (auto i : gammIndex)
                {
                    gammX[j++] = x[i];
                }
                sum += gamm->lf(gammX.data());
            }
            return sum;
            break;
        }
        UMUQFAIL("Unknown Prior type!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

} // namespace umuq

#endif // UMUQ_PRIORDISTRIBUTION
