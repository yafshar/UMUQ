
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
     * \param num_metrics_   Number of metrics (default  1)
     * \param metric_names_  Names of different metrics (default )
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
	fitness(std::string const &metric = "sum_squared", std::size_t const num_metrics_ = 0, std::vector<std::string> const metric_names_ = {""}) : num_metrics(num_metrics_)
	{
		if (!setMetricName(metric))
		{
			UMUQWARNING("Fitness is unknown : By default it is set to sum_squared!");

			fitnessResidual.set(SquredError);
			errorFit = errorFitSum;
		}
	}

	/*!
     * \brief Set the Metric object name
     * 
     * \param metric  Metric name to set for evaluating the model fitness
     */
	bool setMetricName(std::string const &metric)
	{
		fitnessMetricName = metric;
		if (fitnessMetricName == "sum_squared")
		{
			fitnessResidual.set(SquredError);
			errorFit = errorFitSum;
		}
		else if (fitnessMetricName == "mean_squared")
		{
			fitnessResidual.set(SquredError);
			errorFit = errorFitMean;
		}
		else if (fitnessMetricName == "root_mean_squared")
		{
			fitnessResidual.set(SquredError);
			errorFit = errorFitRootMean;
		}
		else if (fitnessMetricName == "max_squared")
		{
			fitnessResidual.set(SquredError);
			errorFit = errorFitMax;
		}
		else if (fitnessMetricName == "sum_scaled")
		{
			fitnessResidual.set(ScaledError);
			errorFit = errorFitSum;
		}
		else if (fitnessMetricName == "mean_scaled")
		{
			fitnessResidual.set(ScaledError);
			errorFit = errorFitMean;
		}
		else if (fitnessMetricName == "max_scaled")
		{
			fitnessResidual.set(ScaledError);
			errorFit = errorFitMax;
		}
		else if (fitnessMetricName == "sum_abs")
		{
			fitnessResidual.set(AbsoluteError);
			errorFit = errorFitSum;
		}
		else if (fitnessMetricName == "mean_abs")
		{
			fitnessResidual.set(AbsoluteError);
			errorFit = errorFitMean;
		}
		else if (fitnessMetricName == "max_abs")
		{
			fitnessResidual.set(AbsoluteError);
			errorFit = errorFitMax;
		}
		else if (fitnessMetricName == "press")
		{
			// return new PRESSFitness();
		}
		else if (fitnessMetricName == "cv")
		{
			// fitnessResidual.set(SquredError);
			// errorFit = errorFitMean;

			// if (num_metrics > 0)
			// {
			//     metric_names.resize(num_metrics);
			//     std::copy(metric_names_.begin(), metric_names_.end(), metric_names.begin());
			//     metric_values.reset(new T[num_metrics]);
			// }
		}
		else if (fitnessMetricName == "rsquared")
		{
			fitnessResidual.set(SquredError);
			errorFit = errorFitSum;
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
     * \param observations array of observations data
     * \param predictions  array of predicted data
     * \param nSize        Size of the array
     * \param results      Array of results
     * \return true        
     * \return false       If there is not enoug hmemory to continue
     */
	bool computeResiduals(T *observations, T *predictions, int const nSize, T *&results)
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

	bool computeResiduals(T *observations, T const prediction, int const nSize, T *&results)
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
		std::for_each(results, results + nSize, [&](T &r_i) { r_i = fitnessResidual(*o++, prediction); });

		return true;
	}

	/*!
     * \brief Get the Fitness value
     * 
     * \param observations Array of observed data
     * \param predictions  Array of predicted data 
     * \param nSize        Size of the array
     * \return             The Fitness value
     */
	T getFitness(T *observations, T *predictions, int const nSize)
	{
		std::unique_ptr<T[]> r;
		try
		{
			r.reset(new T[nSize]);
		}
		catch (std::bad_alloc &e)
		{
			UMUQFAIL("Failed to allocate memory!");
		}

		if (fitnessMetricName == "rsquared")
		{
			T *results = r.get();

			stats s;

			T favg = s.mean<T, T>(observations, nSize);

			T nomin = computeResiduals(predictions, favg, nSize, results) ? s.sum<T, T>(results, nSize) : throw(std::runtime_error("Error!"));

			T denom = computeResiduals(observations, favg, nSize, results) ? s.sum<T, T>(results, nSize) : throw(std::runtime_error("Error!"));

			return nomin / denom;
		}

		if (num_metrics > 0)
		{
			for (int i = 0; i < num_metrics; i++)
			{
				if (metric_names[i] == "sum_squared")
				{
					fitnessResidual.set(SquredError);
					errorFit = errorFitSum;
				}
				else if (metric_names[i] == "mean_squared")
				{
					fitnessResidual.set(SquredError);
					errorFit = errorFitMean;
				}
				else if (metric_names[i] == "root_mean_squared")
				{
					fitnessResidual.set(SquredError);
					errorFit = errorFitRootMean;
				}
				else if (metric_names[i] == "max_squared")
				{
					fitnessResidual.set(SquredError);
					errorFit = errorFitMax;
				}
				else if (metric_names[i] == "sum_scaled")
				{
					fitnessResidual.set(ScaledError);
					errorFit = errorFitSum;
				}
				else if (metric_names[i] == "mean_scaled")
				{
					fitnessResidual.set(ScaledError);
					errorFit = errorFitMean;
				}
				else if (metric_names[i] == "max_scaled")
				{
					fitnessResidual.set(ScaledError);
					errorFit = errorFitMax;
				}
				else if (metric_names[i] == "sum_abs")
				{
					fitnessResidual.set(AbsoluteError);
					errorFit = errorFitSum;
				}
				else if (metric_names[i] == "mean_abs")
				{
					fitnessResidual.set(AbsoluteError);
					errorFit = errorFitMean;
				}
				else if (metric_names[i] == "max_abs")
				{
					fitnessResidual.set(AbsoluteError);
					errorFit = errorFitMax;
				}
				else
				{
					UMUQFAILRETURN("Unknown fitness type!");
				}

				T *results = r.get();
				if (computeResiduals(observations, predictions, nSize, results))
				{
					stats s;
					switch (errorFit)
					{
					case errorFitSum:
						metric_values[i] = s.sum<T, T>(results, nSize);
						break;
					case errorFitMean:
						metric_values[i] = s.mean<T, T>(results, nSize);
						break;
					case errorFitRootMean:
						metric_values[i] = std::sqrt(s.mean<T, T>(results, nSize));
						break;
					case errorFitMax:
						metric_values[i] = s.maxelement<T>(results, nSize);
						break;
					}
				}
				else
				{
					metric_values[i] = std::numeric_limits<T>::max();
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

				switch (errorFit)
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
     * \return std::string 
     */
	std::string getMetricName() { return fitnessMetricName; }

  private:
	//! Fitness metric name
	std::string fitnessMetricName;

	//! Number of metrics
	std::size_t num_metrics;

	//! Rediual type for computing the fitness
	residual<T> fitnessResidual;

	//! Type of error fitness
	int errorFit;

	//! Names of metrics
	std::vector<std::string> metric_names;

	//! Computed fitness values for each metric
	std::unique_ptr<T[]> metric_values;
};

#endif //UMUQ_FITNESS_H
