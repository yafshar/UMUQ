
#ifndef UMUQ_FITNESS_H
#define UMUQ_FITNESS_H

#include "../misc/parser.hpp"
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
enum class ErrorFitnessTypes
{
    /*! Sum of the absolute difference between observed and predicted data. */
    errorFitSum,
    /*! Average of the absolute difference between observed and predicted data. */
    errorFitMean,
    /*! Squared root of the average of the absolute difference between observed and predicted data. */
    errorFitRootMean,
    /*! Maximum value of the absolute difference between observed and predicted data. */
    errorFitMax
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
template <typename DataType>
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
    bool computeResiduals(DataType const *observations, DataType const *predictions, int const nSize, DataType *&results);

    /*!
     * \brief Compute residuals  
     * 
     * \param observations  Array of observations data
     * \param predictions   Array of predicted data
     * \param results       Array of the results
     *
     * \return false If observation and prediction data do not have the same size
     */
    bool computeResiduals(std::vector<DataType> const &observations, std::vector<DataType> const &predictions, std::vector<DataType> &results);

    /*!
     * \brief Compute residuals  
     * 
     * \param observationspredictions  Array of observations & prediction data
     * \param nSize                    Size of the array
     * \param results                  Array of the results
     * 
     * \returns false If there is not enough memory to continue
     */
    bool computeResiduals(DataType const *observationspredictions, int const nSize, DataType *&results);

    /*!
     * \brief Compute residuals  
     * 
     * \param observationspredictions  Array of observations & prediction data
     * \param results                  Array of the results
     * 
     * \returns false If there is not enough memory to continue
     */
    bool computeResiduals(std::vector<DataType> const &observationspredictions, std::vector<DataType> &results);

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
    bool computeResiduals(DataType const *observations, DataType const prediction, int const nSize, DataType *&results);

    /*!
     * \brief Compute residuals
     * 
     * \param observations  Array of observations data
     * \param prediction    A predicted data
     * \param results       Array of the results
     * 
     * \returns false If there is not enough memory to continue
     */
    bool computeResiduals(std::vector<DataType> const &observations, DataType const prediction, std::vector<DataType> &results);

    /*!
     * \brief Get the Fitness value
     * 
     * \param observations  Array of observed data
     * \param predictions   Array of predicted data 
     * \param nSize         Size of the array
     * 
     * \returns The Fitness value
     */
    DataType getFitness(DataType const *observations, DataType const *predictions, int const nSize);

    /*!
     * \brief Get the Fitness value
     * 
     * \param observations  Array of observed data
     * \param predictions   Array of predicted data 
     * 
     * \returns The Fitness value
     */
    DataType getFitness(std::vector<DataType> const &observations, std::vector<DataType> const &predictions);

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
    fitness(fitness<DataType> const &) = delete;

    /*!
     * \brief Delete a fitness object assignment
     * 
     * Make it nonassignable
     * 
     * \returns fitness<DataType>& 
     */
    fitness<DataType> &operator=(fitness<DataType> const &) = delete;

  private:
    //! Fitness metric name
    std::string fitnessMetricName;

    //! Number of metrics
    int numMetrics;

    //! Residual type for computing the fitness
    residual<DataType> fitnessResidual;

    //! Type of error fitness
    ErrorFitnessTypes errorFit;

    //! Names of metrics
    std::vector<std::string> metricNames;

    //! Computed fitness values for each metric
    std::unique_ptr<DataType[]> metricValues;
};

template <typename DataType>
fitness<DataType>::fitness(std::string const &MetricName, int const NumMetrics, std::vector<std::string> const MetricNames) : numMetrics(NumMetrics)
{
    if (!setMetricName(MetricName))
    {
        UMUQWARNING("Fitness is unknown : By default it is set to sum_squared!");

        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
}

template <typename DataType>
bool fitness<DataType>::setMetricName(std::string const &MetricName)
{
    {
        umuq::parser p;
        fitnessMetricName = p.toupper(MetricName);
    }
    if (fitnessMetricName == "SUM_SQUARED")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
    else if (fitnessMetricName == "MEAN_SQUARED")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitMean;
    }
    else if (fitnessMetricName == "ROOT_MEAN_SQUARED")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitRootMean;
    }
    else if (fitnessMetricName == "MAX_SQUARED")
    {
        fitnessResidual.set(ErrorTypes::SquaredError);
        errorFit = ErrorFitnessTypes::errorFitMax;
    }
    else if (fitnessMetricName == "SUM_SCALED")
    {
        fitnessResidual.set(ErrorTypes::ScaledError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
    else if (fitnessMetricName == "MEAN_SCALED")
    {
        fitnessResidual.set(ErrorTypes::ScaledError);
        errorFit = ErrorFitnessTypes::errorFitMean;
    }
    else if (fitnessMetricName == "MAX_SCALED")
    {
        fitnessResidual.set(ErrorTypes::ScaledError);
        errorFit = ErrorFitnessTypes::errorFitMax;
    }
    else if (fitnessMetricName == "SUM_ABS")
    {
        fitnessResidual.set(ErrorTypes::AbsoluteError);
        errorFit = ErrorFitnessTypes::errorFitSum;
    }
    else if (fitnessMetricName == "MEAN_ABS")
    {
        fitnessResidual.set(ErrorTypes::AbsoluteError);
        errorFit = ErrorFitnessTypes::errorFitMean;
    }
    else if (fitnessMetricName == "MAX_ABS")
    {
        fitnessResidual.set(ErrorTypes::AbsoluteError);
        errorFit = ErrorFitnessTypes::errorFitMax;
    }
    else if (fitnessMetricName == "PRESS")
    {
        // return new PRESSFitness();
    }
    else if (fitnessMetricName == "CV")
    {
        // fitnessResidual.set(ErrorTypes::SquaredError);
        // errorFit = errorFitMean;

        // if (numMetrics > 0)
        // {
        //     metricNames.resize(numMetrics);
        //     std::copy(MetricNames.begin(), MetricNames.end(), metricNames.begin());
        //     metricValues.reset(new DataType[numMetrics]);
        // }
    }
    else if (fitnessMetricName == "RSQUARED")
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

template <typename DataType>
bool fitness<DataType>::computeResiduals(DataType const *observations, DataType const *predictions, int const nSize, DataType *&results)
{
    if (results == nullptr)
    {
        try
        {
            results = new DataType[nSize];
        }
        catch (std::bad_alloc &e)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    DataType *o = const_cast<DataType *>(observations);
    DataType *p = const_cast<DataType *>(predictions);
    std::for_each(results, results + nSize, [&](DataType &r_i) { r_i = fitnessResidual(*o++, *p++); });

    return true;
}

template <typename DataType>
bool fitness<DataType>::computeResiduals(std::vector<DataType> const &observations, std::vector<DataType> const &predictions, std::vector<DataType> &results)
{
    if (observations.size() != predictions.size())
    {
        UMUQFAILRETURN("Observation and prediction data does not have the same size!");
    }

    if (results.size() != observations.size())
    {
        results.resize(observations.size());
    }

    auto o = observations.begin();
    auto p = predictions.begin();
    std::for_each(results.begin(), results.end(), [&](DataType &r_i) { r_i = fitnessResidual(*o++, *p++); });

    return true;
}

template <typename DataType>
bool fitness<DataType>::computeResiduals(DataType const *observationspredictions, int const nSize, DataType *&results)
{
    if (nSize & 1)
    {
        UMUQFAILRETURN("Wrong input size!");
    }

    if (results == nullptr)
    {
        try
        {
            results = new DataType[nSize / 2];
        }
        catch (std::bad_alloc &e)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    DataType *o = const_cast<DataType *>(observationspredictions);
    std::for_each(results, results + nSize / 2, [&](DataType &r_i) { r_i = fitnessResidual(*o, *(o + 1)); o += 2; });

    return true;
}

template <typename DataType>
bool fitness<DataType>::computeResiduals(std::vector<DataType> const &observationspredictions, std::vector<DataType> &results)
{
    if (observationspredictions.size() & 1)
    {
        UMUQFAILRETURN("Wrong input size!");
    }

    if (results.size() != observationspredictions.size() / 2)
    {
        results.resize(observationspredictions.size() / 2);
    }

    auto o = observationspredictions.begin();
    std::for_each(results.begin(), results.end(), [&](DataType &r_i) { r_i = fitnessResidual(*o, *(o + 1)); o += 2; });

    return true;
}

template <typename DataType>
bool fitness<DataType>::computeResiduals(DataType const *observations, DataType const prediction, int const nSize, DataType *&results)
{
    if (results == nullptr)
    {
        try
        {
            results = new DataType[nSize];
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    DataType *o = const_cast<DataType *>(observations);
    std::for_each(results, results + nSize, [&](DataType &r_i) { r_i = fitnessResidual(*o++, prediction); });

    return true;
}

template <typename DataType>
bool fitness<DataType>::computeResiduals(std::vector<DataType> const &observations, DataType const prediction, std::vector<DataType> &results)
{
    if (results.size() != observations.size())
    {
        results.resize(observations.size());
    }

    auto o = observations.begin();
    std::for_each(results.begin(), results.end(), [&](DataType &r_i) { r_i = fitnessResidual(*o++, prediction); });

    return true;
}

template <typename DataType>
DataType fitness<DataType>::getFitness(DataType const *observations, DataType const *predictions, int const nSize)
{
    std::vector<DataType> r(nSize);

    if (fitnessMetricName == "RSQUARED")
    {
        DataType *results = r.data();
        umuq::stats s;
        DataType observationsAvg = s.mean<DataType, DataType>(observations, nSize);
        DataType rsqNominator = computeResiduals(predictions, observationsAvg, nSize, results) ? s.sum<DataType, DataType>(results, nSize) : throw(std::runtime_error("Error!"));
        DataType rsqDenominator = computeResiduals(observations, observationsAvg, nSize, results) ? s.sum<DataType, DataType>(results, nSize) : throw(std::runtime_error("Error!"));
        return rsqNominator / rsqDenominator;
    }

    if (numMetrics > 0)
    {
        {
            umuq::parser p;
            for (int i = 0; i < numMetrics; i++)
            {
                metricNames[i] = p.toupper(metricNames[i]);
            }
        }
        for (int i = 0; i < numMetrics; i++)
        {
            if (metricNames[i] == "SUM_SQUARED")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "MEAN_SQUARED")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "ROOT_MEAN_SQUARED")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitRootMean;
            }
            else if (metricNames[i] == "MAX_SQUARED")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else if (metricNames[i] == "SUM_SCALED")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "MEAN_SCALED")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "MAX_SCALED")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else if (metricNames[i] == "SUM_ABS")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "MEAN_ABS")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "MAX_ABS")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else
            {
                UMUQFAILRETURN("Unknown fitness type!");
            }

            DataType *results = r.data();
            if (computeResiduals(observations, predictions, nSize, results))
            {
                umuq::stats s;
                switch (errorFit)
                {
                case ErrorFitnessTypes::errorFitSum:
                    metricValues[i] = s.sum<DataType, DataType>(results, nSize);
                    break;
                case ErrorFitnessTypes::errorFitMean:
                    metricValues[i] = s.mean<DataType, DataType>(results, nSize);
                    break;
                case ErrorFitnessTypes::errorFitRootMean:
                    metricValues[i] = std::sqrt(s.mean<DataType, DataType>(results, nSize));
                    break;
                case ErrorFitnessTypes::errorFitMax:
                    metricValues[i] = s.maxelement<DataType>(results, nSize);
                    break;
                }
            }
            else
            {
                metricValues[i] = std::numeric_limits<DataType>::max();
            }
        }
        return DataType{};
    }
    else
    {
        DataType *results = r.data();
        if (computeResiduals(observations, predictions, nSize, results))
        {
            umuq::stats s;

            switch (errorFit)
            {
            case ErrorFitnessTypes::errorFitSum:
                return s.sum<DataType, DataType>(results, nSize);
            case ErrorFitnessTypes::errorFitMean:
                return s.mean<DataType, DataType>(results, nSize);
            case ErrorFitnessTypes::errorFitRootMean:
                return std::sqrt(s.mean<DataType, DataType>(results, nSize));
            case ErrorFitnessTypes::errorFitMax:
                return s.maxelement<DataType>(results, nSize);
            }
        }
    }
    return std::numeric_limits<DataType>::max();
}

template <typename DataType>
DataType fitness<DataType>::getFitness(std::vector<DataType> const &observations, std::vector<DataType> const &predictions)
{
    if (observations.size() != predictions.size())
    {
        UMUQFAIL("Observation and prediction data does not have the same size!");
    }

    std::vector<DataType> results(observations.size());

    if (fitnessMetricName == "RSQUARED")
    {
        umuq::stats s;
        DataType const observationsAvg = s.mean<DataType, DataType>(observations);
        DataType const rsqNominator = computeResiduals(predictions, observationsAvg, results) ? s.sum<DataType, DataType>(results) : throw(std::runtime_error("Error!"));
        DataType const rsqDenominator = computeResiduals(observations, observationsAvg, results) ? s.sum<DataType, DataType>(results) : throw(std::runtime_error("Error!"));
        return rsqNominator / rsqDenominator;
    }

    if (numMetrics > 0)
    {
        {
            umuq::parser p;
            for (int i = 0; i < numMetrics; i++)
            {
                metricNames[i] = p.toupper(metricNames[i]);
            }
        }
        for (int i = 0; i < numMetrics; i++)
        {
            if (metricNames[i] == "SUM_SQUARED")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "MEAN_SQUARED")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "ROOT_MEAN_SQUARED")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitRootMean;
            }
            else if (metricNames[i] == "MAX_SQUARED")
            {
                fitnessResidual.set(ErrorTypes::SquaredError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else if (metricNames[i] == "SUM_SCALED")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "MEAN_SCALED")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "MAX_SCALED")
            {
                fitnessResidual.set(ErrorTypes::ScaledError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else if (metricNames[i] == "SUM_ABS")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitSum;
            }
            else if (metricNames[i] == "MEAN_ABS")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitMean;
            }
            else if (metricNames[i] == "MAX_ABS")
            {
                fitnessResidual.set(ErrorTypes::AbsoluteError);
                errorFit = ErrorFitnessTypes::errorFitMax;
            }
            else
            {
                UMUQFAILRETURN("Unknown fitness type!");
            }

            if (computeResiduals(observations, predictions, results))
            {
                umuq::stats s;
                switch (errorFit)
                {
                case ErrorFitnessTypes::errorFitSum:
                    metricValues[i] = s.sum<DataType, DataType>(results);
                    break;
                case ErrorFitnessTypes::errorFitMean:
                    metricValues[i] = s.mean<DataType, DataType>(results);
                    break;
                case ErrorFitnessTypes::errorFitRootMean:
                    metricValues[i] = std::sqrt(s.mean<DataType, DataType>(results));
                    break;
                case ErrorFitnessTypes::errorFitMax:
                    metricValues[i] = s.maxelement<DataType>(results);
                    break;
                }
            }
            else
            {
                metricValues[i] = std::numeric_limits<DataType>::max();
            }
        }
        return DataType{};
    }
    else
    {
        if (computeResiduals(observations, predictions, results))
        {
            umuq::stats s;

            switch (errorFit)
            {
            case ErrorFitnessTypes::errorFitSum:
                return s.sum<DataType, DataType>(results);
            case ErrorFitnessTypes::errorFitMean:
                return s.mean<DataType, DataType>(results);
            case ErrorFitnessTypes::errorFitRootMean:
                return std::sqrt(s.mean<DataType, DataType>(results));
            case ErrorFitnessTypes::errorFitMax:
                return s.maxelement<DataType>(results);
            }
        }
    }
    return std::numeric_limits<DataType>::max();
}

template <typename DataType>
inline std::string fitness<DataType>::getMetricName() { return fitnessMetricName; }

} // namespace umuq

#endif // UMUQ_FITNESS
