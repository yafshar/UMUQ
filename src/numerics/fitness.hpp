
#ifndef UMUQ_FITNESS_H
#define UMUQ_FITNESS_H

#include "residual.hpp"
#include "stats.hpp"

enum
{
    errorFitSum = -1,
    errorFitMean = -2,
    errorFitRootMean = -3,
    errorFitMax = -4
};

/*! \class fitness
 * \ingroup numerics
 *
 * \brief This class evalutes the model fitness 
 *
 * List of available Fitness type:
 *  - \b errorFitSum      Sum of the absolute difference between observed and predicted data
 *  - \b errorFitMean     Average of the absolute difference between observed and predicted data
 *  - \b errorFitRootMean Squared root of the average of the absolute difference between observed and predicted data
 *  - \b errorFitMax      Maximum value of the absolute difference between observed and predicted data
 */
template <typename T>
class fitness
{
  public:
    /*!
     * \brief Construct a new fitness object
     * 
     * \param metric         Metric name for evaluating the model fitness
     * \param inum_metrics   Input number of metrics (default  0)
     * \param imetric_names  Input names of different metrics (default )
     * 
     * List of available metrics:
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
    fitness(std::string const &metric = "sum_squared", int const inum_metrics = 0, std::vector<std::string> const imetric_names = {""});

    /*!
     * \brief Set the Metric object name
     * 
     * \param metric  Metric name to set for evaluating the model fitness
     */
    bool setMetricName(std::string const &metric);

    /*!
     * \brief Compute residuals  
     * 
     * \param observations array of observations data
     * \param predictions  array of predicted data
     * \param nSize        Size of the array
     * \param results      Array of results
     * \return true        
     * \return false       If there is not enoug hmemory to continue
     */
    bool computeResiduals(T *observations, T *predictions, int const nSize, T *&results);

    bool computeResiduals(T *observations, T const prediction, int const nSize, T *&results);

    /*!
     * \brief Get the Fitness value
     * 
     * \param observations Array of observed data
     * \param predictions  Array of predicted data 
     * \param nSize        Size of the array
     * \return             The Fitness value
     */
    T getFitness(T *observations, T *predictions, int const nSize);

    /*!
     * \brief Get the Metric object name
     * 
     * \return std::string 
     */
    inline std::string getMetricName();

  private:
    //! Fitness metric name
    std::string fitnessMetricName;

    //! Number of metrics
    int num_metrics;

    //! Rediual type for computing the fitness
    residual<T> fitnessResidual;

    //! Type of error fitness
    int errorFit;

    //! Names of metrics
    std::vector<std::string> metric_names;

    //! Computed fitness values for each metric
    std::unique_ptr<T[]> metric_values;
};

/*!
 * \brief Construct a new fitness object
 * 
 * \param metric         Metric name for evaluating the model fitness
 * \param inum_metrics   Input number of metrics (default  1)
 * \param imetric_names  Input names of different metrics (default )
 * 
 * List of available metrics:
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
template <typename T>
fitness<T>::fitness(std::string const &metric, int const inum_metrics, std::vector<std::string> const imetric_names) : num_metrics(inum_metrics)
{
    if (!this->setMetricName(metric))
    {
        UMUQWARNING("Fitness is unknown : By default it is set to sum_squared!");

        this->fitnessResidual.set(SquredError);
        this->errorFit = errorFitSum;
    }
}

/*!
 * \brief Set the Metric object name
 * 
 * \param metric  Metric name to set for evaluating the model fitness
 */
template <typename T>
bool fitness<T>::setMetricName(std::string const &metric)
{
    this->fitnessMetricName = metric;
    if (this->fitnessMetricName == "sum_squared")
    {
        this->fitnessResidual.set(SquredError);
        this->errorFit = errorFitSum;
    }
    else if (this->fitnessMetricName == "mean_squared")
    {
        this->fitnessResidual.set(SquredError);
        this->errorFit = errorFitMean;
    }
    else if (this->fitnessMetricName == "root_mean_squared")
    {
        this->fitnessResidual.set(SquredError);
        this->errorFit = errorFitRootMean;
    }
    else if (this->fitnessMetricName == "max_squared")
    {
        this->fitnessResidual.set(SquredError);
        this->errorFit = errorFitMax;
    }
    else if (this->fitnessMetricName == "sum_scaled")
    {
        this->fitnessResidual.set(ScaledError);
        this->errorFit = errorFitSum;
    }
    else if (this->fitnessMetricName == "mean_scaled")
    {
        this->fitnessResidual.set(ScaledError);
        this->errorFit = errorFitMean;
    }
    else if (this->fitnessMetricName == "max_scaled")
    {
        this->fitnessResidual.set(ScaledError);
        this->errorFit = errorFitMax;
    }
    else if (this->fitnessMetricName == "sum_abs")
    {
        this->fitnessResidual.set(AbsoluteError);
        this->errorFit = errorFitSum;
    }
    else if (this->fitnessMetricName == "mean_abs")
    {
        this->fitnessResidual.set(AbsoluteError);
        this->errorFit = errorFitMean;
    }
    else if (this->fitnessMetricName == "max_abs")
    {
        this->fitnessResidual.set(AbsoluteError);
        this->errorFit = errorFitMax;
    }
    else if (this->fitnessMetricName == "press")
    {
        // return new PRESSFitness();
    }
    else if (this->fitnessMetricName == "cv")
    {
        // this->fitnessResidual.set(SquredError);
        // this->errorFit = errorFitMean;

        // if (this->num_metrics > 0)
        // {
        //     this->metric_names.resize(this->num_metrics);
        //     std::copy(imetric_names.begin(), imetric_names.end(), this->metric_names.begin());
        //     this->metric_values.reset(new T[this->num_metrics]);
        // }
    }
    else if (this->fitnessMetricName == "rsquared")
    {
        this->fitnessResidual.set(SquredError);
        this->errorFit = errorFitSum;
    }
    else
    {
        return false;
    }
    return true;
}

/*!
 * \brief Compute residuals  
 * 
 * \param observations  Array of observations data
 * \param predictions   Array of predicted data
 * \param nSize         Size of the array
 * \param results       Array of results
 * 
 * \return true        
 * \return false        If there is not enoug hmemory to continue
 */
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
    std::for_each(results, results + nSize, [&](T &r_i) { r_i = this->fitnessResidual(*o++, *p++); });

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
    std::for_each(results, results + nSize, [&](T &r_i) { r_i = this->fitnessResidual(*o++, prediction); });

    return true;
}

/*!
 * \brief Get the Fitness value
 * 
 * \param observations Array of observed data
 * \param predictions  Array of predicted data 
 * \param nSize        Size of the array
 * 
 * \return             The Fitness value
 */
template <typename T>
T fitness<T>::getFitness(T *observations, T *predictions, int const nSize)
{
    std::unique_ptr<T[]> r;
    try
    {
        r.reset(new T[nSize]);
    }
    catch (...)
    {
        UMUQFAIL("Failed to allocate memory!");
    }

    if (this->fitnessMetricName == "rsquared")
    {
        T *results = r.get();

        stats s;

        T favg = s.mean<T, T>(observations, nSize);

        T nomin = computeResiduals(predictions, favg, nSize, results) ? s.sum<T, T>(results, nSize) : throw(std::runtime_error("Error!"));

        T denom = computeResiduals(observations, favg, nSize, results) ? s.sum<T, T>(results, nSize) : throw(std::runtime_error("Error!"));

        return nomin / denom;
    }

    if (this->num_metrics > 0)
    {
        for (int i = 0; i < this->num_metrics; i++)
        {
            if (this->metric_names[i] == "sum_squared")
            {
                this->fitnessResidual.set(SquredError);
                this->errorFit = errorFitSum;
            }
            else if (this->metric_names[i] == "mean_squared")
            {
                this->fitnessResidual.set(SquredError);
                this->errorFit = errorFitMean;
            }
            else if (this->metric_names[i] == "root_mean_squared")
            {
                this->fitnessResidual.set(SquredError);
                this->errorFit = errorFitRootMean;
            }
            else if (this->metric_names[i] == "max_squared")
            {
                this->fitnessResidual.set(SquredError);
                this->errorFit = errorFitMax;
            }
            else if (this->metric_names[i] == "sum_scaled")
            {
                this->fitnessResidual.set(ScaledError);
                this->errorFit = errorFitSum;
            }
            else if (this->metric_names[i] == "mean_scaled")
            {
                this->fitnessResidual.set(ScaledError);
                this->errorFit = errorFitMean;
            }
            else if (this->metric_names[i] == "max_scaled")
            {
                this->fitnessResidual.set(ScaledError);
                this->errorFit = errorFitMax;
            }
            else if (this->metric_names[i] == "sum_abs")
            {
                this->fitnessResidual.set(AbsoluteError);
                this->errorFit = errorFitSum;
            }
            else if (this->metric_names[i] == "mean_abs")
            {
                this->fitnessResidual.set(AbsoluteError);
                this->errorFit = errorFitMean;
            }
            else if (this->metric_names[i] == "max_abs")
            {
                this->fitnessResidual.set(AbsoluteError);
                this->errorFit = errorFitMax;
            }
            else
            {
                UMUQFAILRETURN("Unknown fitness type!");
            }

            T *results = r.get();
            if (computeResiduals(observations, predictions, nSize, results))
            {
                stats s;
                switch (this->errorFit)
                {
                case errorFitSum:
                    this->metric_values[i] = s.sum<T, T>(results, nSize);
                    break;
                case errorFitMean:
                    this->metric_values[i] = s.mean<T, T>(results, nSize);
                    break;
                case errorFitRootMean:
                    this->metric_values[i] = std::sqrt(s.mean<T, T>(results, nSize));
                    break;
                case errorFitMax:
                    this->metric_values[i] = s.maxelement<T>(results, nSize);
                    break;
                }
            }
            else
            {
                this->metric_values[i] = std::numeric_limits<T>::max();
            }
        }
        return T{};
    }
    else
    {
        T *results = r.get();
        if (computeResiduals(observations, predictions, nSize, results))
        {
            stats s;

            switch (this->errorFit)
            {
            case errorFitSum:
                return s.sum<T, T>(results, nSize);
            case errorFitMean:
                return s.mean<T, T>(results, nSize);
            case errorFitRootMean:
                return std::sqrt(s.mean<T, T>(results, nSize));
            case errorFitMax:
                return s.maxelement<T>(results, nSize);
            }
        }
    }
    return std::numeric_limits<T>::max();
}

/*!
 * \brief Get the Metric object name
 * 
 * \return the Metric object name
 */
template <typename T>
inline std::string fitness<T>::getMetricName() { return this->fitnessMetricName; }

#endif //UMUQ_FITNESS_H
