
#ifndef UMUQ_FITNESS_H
#define UMUQ_FITNESS_H

#include "residual.hpp"
#include "stats.hpp"

namespace umuq
{

/*! \enum ErrorFitnessTypes
 * \ingroup Numerics_Module
 * 
 * \brief Available Error Fitness type, currently available in %UMUQ
 * 
 */
enum ErrorFitnessTypes
{
    /*! Sum of the absolute difference between observed and predicted data. */
    errorFitSum = -1,
    /*! Average of the absolute difference between observed and predicted data. */
    errorFitMean = -2,
    /*! Squared root of the average of the absolute difference between observed and predicted data. */
    errorFitRootMean = -3,
    /*! Maximum value of the absolute difference between observed and predicted data. */
    errorFitMax = -4
};

/*! \class fitness
 * \ingroup Numerics_Module
 *
 * \brief This class evalutes the model fitness 
 *
 * List of available Fitness type:
 *  - \b errorFitSum      Sum of the absolute difference between observed and predicted data.
 *  - \b errorFitMean     Average of the absolute difference between observed and predicted data.
 *  - \b errorFitRootMean Squared root of the average of the absolute difference between observed and predicted data.
 *  - \b errorFitMax      Maximum value of the absolute difference between observed and predicted data.
 */
template <typename T>
class fitness
{
  public:
    /*!
     * \brief Construct a new fitness object
     * 
     * \param MetricName   Metric name for evaluating the model fitness
     * \param NumMetrics   Number of metrics (default  0)
     * \param MetricNames  Names of different metrics (default )
     * 
     * List of available metrics:<br>
     *  - \b sum_squared
     *  - \b mean_squared     
     *  - \b root_mean_squared     
     *  - \b max_squared
     *  - \b sum_scaled
     *  - \b mean_scaled     
     *  - \b root_mean_scaled    
     *  - \b max_scaled
     *  - \b sum_abs
     *  - \b mean_abs   
     *  - \b root_mean_abs
     *  - \b max_abs
     *  - \b press   
     *  - \b cv
     *  - \b rsquared
     */
    fitness(std::string const &MetricName = "sum_squared", int const NumMetrics = 0, std::vector<std::string> const MetricNames = {""});

    /*!
     * \brief Set the Metric object name
     * 
     * \param MetricName  Metric name to set for evaluating the model fitness
     */
    bool setMetricName(std::string const &MetricName);

    /*!
     * \brief Compute residuals  
     * 
     * \param observations  Array of observations data
     * \param predictions   Array of predicted data
     * \param nSize         Size of the array
     * \param results       Array of the results
     *
     * \return false If there is not enough memory to continue
     */
    bool computeResiduals(T *observations, T *predictions, int const nSize, T *&results);

    /*!
     * \brief Compute residuals  
     * 
     * \param observationspredictions  Array of observations & prediction data
     * \param nSize                    Size of the array
     * \param results                  Array of the results
     * 
     * \returns false If there is not enough memory to continue
     */
    bool computeResiduals(T *observationspredictions, int const nSize, T *&results);

    /*!
     * \brief Compute residuals
     * 
     * \param observations  Array of observations data
     * \param prediction    A predicted data
     * \param nSize         Size of the array
     * \param results       Array of the results
     * 
     * \returns false If there is not enough memory to continue
     */
    bool computeResiduals(T *observations, T const prediction, int const nSize, T *&results);

    /*!
     * \brief Get the Fitness value
     * 
     * \param observations  Array of observed data
     * \param predictions   Array of predicted data 
     * \param nSize         Size of the array
     * 
     * \returns The Fitness value
     */
    T getFitness(T *observations, T *predictions, int const nSize);

    /*!
     * \brief Get the Metric object name
     * 
     * \return std::string The name of the Metric object
     */
    inline std::string getMetricName();

  protected:
    /*!
     * \brief Delete a fitness object copy construction
     * 
     * Make it noncopyable.
     */
    fitness(fitness<T> const &) = delete;

    /*!
     * \brief Delete a fitness object assignment
     * 
     * Make it nonassignable
     * 
     * \returns fitness<T>& 
     */
    fitness<T> &operator=(fitness<T> const &) = delete;

  private:
    //! Fitness metric name
    std::string fitnessMetricName;

    //! Number of metrics
    int numMetrics;

    //! Residual type for computing the fitness
    residual<T> fitnessResidual;

    //! Type of error fitness
    int errorFit;

    //! Names of metrics
    std::vector<std::string> metricNames;

    //! Computed fitness values for each metric
    std::unique_ptr<T[]> metricValues;
};

template <typename T>
fitness<T>::fitness(std::string const &MetricName, int const NumMetrics, std::vector<std::string> const MetricNames) : numMetrics(NumMetrics)
{
    if (!setMetricName(MetricName))
    {
        UMUQWARNING("Fitness is unknown : By default it is set to sum_squared!");

        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
}

template <typename T>
bool fitness<T>::setMetricName(std::string const &MetricName)
{
    fitnessMetricName = MetricName;
    if (fitnessMetricName == "sum_squared")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
    else if (fitnessMetricName == "mean_squared")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitMean;
    }
    else if (fitnessMetricName == "root_mean_squared")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitRootMean;
    }
    else if (fitnessMetricName == "max_squared")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitMax;
    }
    else if (fitnessMetricName == "sum_scaled")
    {
        fitnessResidual.set(ErrorTypes::ScaledError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
    else if (fitnessMetricName == "mean_scaled")
    {
        fitnessResidual.set(ErrorTypes::ScaledError);
        errorFit = ErrorFitnessTypes::errorFitMean;
    }
    else if (fitnessMetricName == "max_scaled")
    {
        fitnessResidual.set(ErrorTypes::ScaledError);
        errorFit = ErrorFitnessTypes::errorFitMax;
    }
    else if (fitnessMetricName == "sum_abs")
    {
        fitnessResidual.set(ErrorTypes::AbsoluteError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
    else if (fitnessMetricName == "mean_abs")
    {
        fitnessResidual.set(ErrorTypes::AbsoluteError);
        errorFit = ErrorFitnessTypes::errorFitMean;
    }
    else if (fitnessMetricName == "max_abs")
    {
        fitnessResidual.set(ErrorTypes::AbsoluteError);
        errorFit = ErrorFitnessTypes::errorFitMax;
    }
    else if (fitnessMetricName == "press")
    {
        // return new PRESSFitness();
    }
    else if (fitnessMetricName == "cv")
    {
        // fitnessResidual.set(ErrorTypes::SquaredError);
        // errorFit = errorFitMean;

        // if (numMetrics > 0)
        // {
        //     metricNames.resize(numMetrics);
        //     std::copy(MetricNames.begin(), MetricNames.end(), metricNames.begin());
        //     metricValues.reset(new T[numMetrics]);
        // }
    }
    else if (fitnessMetricName == "rsquared")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
    else
    {
        return false;
    }
    return true;
}

template <typename T>
bool fitness<T>::computeResiduals(T *observations, T *predictions, int const nSize, T *&results)
{
    if (results == nullptr)
    {
        try
        {
            results = new T[nSize];
        }
        catch (std::bad_alloc &e)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    T *o = observations;
    T *p = predictions;
    std::for_each(results, results + nSize, [&](T &r_i) { r_i = fitnessResidual(*o++, *p++); });

    return true;
}

template <typename T>
bool fitness<T>::computeResiduals(T *observationspredictions, int const nSize, T *&results)
{
    if (nSize % 2 != 0)
    {
        UMUQFAILRETURN("Wrong input size!");
    }

    if (results == nullptr)
    {
        try
        {
            results = new T[nSize / 2];
        }
        catch (std::bad_alloc &e)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    T *o = observationspredictions;
    std::for_each(results, results + nSize / 2, [&](T &r_i) { r_i = fitnessResidual(*o, *(o + 1)); o += 2; });

    return true;
}

template <typename T>
bool fitness<T>::computeResiduals(T *observations, T const prediction, int const nSize, T *&results)
{
    if (results == nullptr)
    {
        try
        {
            results = new T[nSize];
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    T *o = observations;
    std::for_each(results, results + nSize, [&](T &r_i) { r_i = fitnessResidual(*o++, prediction); });

    return true;
}

template <typename T>
T fitness<T>::getFitness(T *observations, T *predictions, int const nSize)
{
    std::vector<T> r(nSize);

    if (fitnessMetricName == "rsquared")
    {
        T *results = r.data();

        umuq::stats s;

        T favg = s.mean<T, T>(observations, nSize);

        T nomin = computeResiduals(predictions, favg, nSize, results) ? s.sum<T, T>(results, nSize) : throw(std::runtime_error("Error!"));

        T denom = computeResiduals(observations, favg, nSize, results) ? s.sum<T, T>(results, nSize) : throw(std::runtime_error("Error!"));

        return nomin / denom;
    }

    if (numMetrics > 0)
    {
        for (int i = 0; i < numMetrics; i++)
        {
            if (metricNames[i] == "sum_squared")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "mean_squared")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "root_mean_squared")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitRootMean;
            }
            else if (metricNames[i] == "max_squared")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else if (metricNames[i] == "sum_scaled")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "mean_scaled")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "max_scaled")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else if (metricNames[i] == "sum_abs")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "mean_abs")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "max_abs")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else
            {
                UMUQFAILRETURN("Unknown fitness type!");
            }

            T *results = r.data();
            if (computeResiduals(observations, predictions, nSize, results))
            {
                umuq::stats s;
                switch (errorFit)
                {
                case ErrorFitnessTypes::errorFitSum:
                    metricValues[i] = s.sum<T, T>(results, nSize);
                    break;
                case ErrorFitnessTypes::errorFitMean:
                    metricValues[i] = s.mean<T, T>(results, nSize);
                    break;
                case ErrorFitnessTypes::errorFitRootMean:
                    metricValues[i] = std::sqrt(s.mean<T, T>(results, nSize));
                    break;
                case ErrorFitnessTypes::errorFitMax:
                    metricValues[i] = s.maxelement<T>(results, nSize);
                    break;
                }
            }
            else
            {
                metricValues[i] = std::numeric_limits<T>::max();
            }
        }
        return T{};
    }
    else
    {
        T *results = r.data();
        if (computeResiduals(observations, predictions, nSize, results))
        {
            umuq::stats s;

            switch (errorFit)
            {
            case ErrorFitnessTypes::errorFitSum:
                return s.sum<T, T>(results, nSize);
            case ErrorFitnessTypes::errorFitMean:
                return s.mean<T, T>(results, nSize);
            case ErrorFitnessTypes::errorFitRootMean:
                return std::sqrt(s.mean<T, T>(results, nSize));
            case ErrorFitnessTypes::errorFitMax:
                return s.maxelement<T>(results, nSize);
            }
        }
    }
    return std::numeric_limits<T>::max();
}

template <typename T>
inline std::string fitness<T>::getMetricName() { return fitnessMetricName; }

} // namespace umuq

#endif // UMUQ_FITNESS
