#ifndef UMUQ_STDATA_H
#define UMUQ_STDATA_H

#include "../core/core.hpp"
#include "../io/io.hpp"
#include "../misc/parser.hpp"

/*! \class optimizationParameters
 * \brief This is a class to set the optimization parameters 
 * 
 * \tparam T Data type
 */
template <typename T>
struct optimizationParameters
{
	int MaxIter;
	int Display;
	T Tolerance;
	T Step;

	/*!
     *  \brief Default constructor for the default variables
     *    
     */
	optimizationParameters() : MaxIter(100),
							   Display(0),
							   Tolerance(1e-6),
							   Step(1e-5){};
};

/*! \class stdata
 *  \brief stream data type class
 *
 * \param nDim                       Problem Dimension
 * \param maxGenerations             Maximum number of generations
 * \param populationSize             Sampling population size
 * \param lastPopulationSize         Sampling population size in the final generation
 * \param auxilSize                  Auxillary data size
 * \param minChainLength             Minimum size of the chain in the TMCMC algorithm (default 1)
 * \param maxChainLength             Maximum size of the chain in the TMCMC algorithm (default 1)
 * \param seed                       Random number initial seed
 * \param samplingType               Sampling type which is : 0: uniform, 1: gaussian, 2: file
 * \param priorType                  Prior type which is :   0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4: composit
 * \param iPlot                      1 for printing the data and 0 for not
 * \param saveData                   1 for saving the data and 0 for not
 * \param useCmaProposal             Indicator if we use the CMA proposal or not
 * \param useLocalCovariance         Indicator if we use the local covariance or not
 * \param lb                         Generic lower bound (It is -6 per default)
 * \param ub                         Generic upper bound (It is 6 per default)
 * \param TolCOV                     A prescribed tolerance
 * \param bbeta                      \f$ \beta \f$ parameter in the TMCMC algorithm
 * \param localScale                 Local scale
 * \param options                    Optimization parameter
 * \param eachPopulationSize         Sampling population size for each generation
 * \param lowerBound                 Sampling domain lower bounds for each dimension
 * \param upperBound                 Sampling domain upper bounds for each dimension
 * \param compositePriorDistribution Composite distribution as a prior
 * \param priorMu                    Prior mean, in case of gamma distribution it is alpha
 * \param priorSigma                 Prior standard deviation 
 * \param auxilData                  Auxillary data
 * \param initMean                   Initial Mean with the size of [populationSize*nDim]
 * \param localCovariance            Local covariance with the size of [populationSize*nDim*nDim]
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
	 * \return stdata<T>& 
	 */
	stdata<T> &operator=(stdata<T> &&other);

	/*!
	 * \brief reset the stream data values to the input values
	 * 
	 * \param probdim          Problem Dimension
	 * \param MaxGenerations   Maximum number of generations
	 * \param PopulationSize   Sampling population size
	 * 
	 * \return true 
	 * \return false If there is not enough memory available for allocating the data
	 */
	bool reset(int probdim, int MaxGenerations, int PopulationSize);

	/*!
     * \brief load the input file fname
     *
     * \param fname              name of the input file
     * \return true on success
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
	// Make it noncopyable
	stdata(stdata<T> const &) = delete;

	// Make it not assignable
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
	//! A prescribed tolerance
	T TolCOV;

	//! \f$ \beta \f$ parameter in the TMCMC algorithm
	T bbeta;

	//! Local scale
	T localScale;

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

	//! Initial Mean with the size of [populationSize*nDim]
	std::vector<T> initMean;

	//! Local covariance with the size of [populationSize*nDim*nDim]
	std::vector<T> localCovariance;
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
					  TolCOV(static_cast<T>(1)),
					  bbeta(static_cast<T>(0.2)),
					  localScale(0),
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
																		 TolCOV(static_cast<T>(1)),
																		 bbeta(static_cast<T>(0.2)),
																		 localScale(0),
																		 options(),
																		 eachPopulationSize(maxGenerations),
																		 lowerBound(nDim, T{}),
																		 upperBound(nDim, T{}),
																		 priorParam1(nDim, T{}),
																		 priorParam2(nDim * nDim, T{}),
																		 localCovariance(populationSize * nDim * nDim, T{})
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

	for (int i = 0, l = 0; i < populationSize; i++)
	{
		for (int j = 0; j < nDim; j++)
		{
			for (int k = 0; k < nDim; k++, l++)
			{
				if (j == k)
				{
					localCovariance[l] = static_cast<T>(1);
				}
			}
		}
	}
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
	TolCOV = other.TolCOV;
	bbeta = other.bbeta;
	localScale = other.localScale;
	options.Display = other.options.Display;
	options.MaxIter = other.options.MaxIter;
	options.Step = other.options.Step;
	options.Tolerance = other.options.Tolerance;
	eachPopulationSize = std::move(other.eachPopulationSize);
	lowerBound = std::move(other.lowerBound);
	upperBound = std::move(other.upperBound);
	compositePriorDistribution = std::move(other.compositePriorDistribution);
	priorParam1 = std::move(other.priorParam1);
	priorParam2 = std::move(other.priorParam2);
	auxilData = std::move(other.auxilData);
	initMean = std::move(other.initMean);
	localCovariance = std::move(other.localCovariance);
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
	TolCOV = other.TolCOV;
	bbeta = other.bbeta;
	localScale = other.localScale;
	options.Display = other.options.Display;
	options.MaxIter = other.options.MaxIter;
	options.Step = other.options.Step;
	options.Tolerance = other.options.Tolerance;
	eachPopulationSize = std::move(other.eachPopulationSize);
	lowerBound = std::move(other.lowerBound);
	upperBound = std::move(other.upperBound);
	compositePriorDistribution = std::move(other.compositePriorDistribution);
	priorParam1 = std::move(other.priorParam1);
	priorParam2 = std::move(other.priorParam2);
	auxilData = std::move(other.auxilData);
	initMean = std::move(other.initMean);
	localCovariance = std::move(other.localCovariance);

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
	std::swap(TolCOV, other.TolCOV);
	std::swap(bbeta, other.bbeta);
	std::swap(localScale, other.localScale);
	std::swap(options.Display, other.options.Display);
	std::swap(options.MaxIter, other.options.MaxIter);
	std::swap(options.Step, other.options.Step);
	std::swap(options.Tolerance, other.options.Tolerance);
	eachPopulationSize.swap(other.eachPopulationSize);
	lowerBound.swap(other.lowerBound);
	upperBound.swap(other.upperBound);
	compositePriorDistribution.swap(other.compositePriorDistribution);
	priorParam1.swap(other.priorParam1);
	priorParam2.swap(other.priorParam2);
	auxilData.swap(other.auxilData);
	initMean.swap(other.initMean);
	localCovariance.swap(other.localCovariance);
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
	TolCOV = static_cast<T>(1);
	bbeta = static_cast<T>(0.2);
	localScale = 0;
	options.MaxIter = 100;
	options.Display = 0;
	options.Tolerance = static_cast<T>(1e-6);
	options.Step = static_cast<T>(1e-5);

	if (probdim == 0 || MaxGenerations == 0 || PopulationSize == 0)
	{
		nDim = 0;
		maxGenerations = 0;
		populationSize = 0;
		lastPopulationSize = 0;

		eachPopulationSize.clear();
		lowerBound.clear();
		upperBound.clear();
		priorParam1.clear();
		priorParam2.clear();
		localCovariance.clear();

		std::cout << "Warning : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cout << " Reseting to size zero! " << std::endl;

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
		localCovariance.resize(populationSize * nDim * nDim, T{});
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

	for (int i = 0, l = 0; i < populationSize; i++)
	{
		for (int j = 0; j < nDim; j++)
		{
			for (int k = 0; k < nDim; k++, l++)
			{
				if (j == k)
				{
					localCovariance[l] = static_cast<T>(1);
				}
			}
		}
	}

	return true;
}

/*!
 * \brief load the input file fname for setting the input variables
 * 
 * \tparam T      Data type
 * 
 * \param fname   Input file name
 *  
 * \return true 
 * \return false 
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

			//! These are temporary variables
			int probdim = nDim;
			int maxgens = maxGenerations;
			int datanum = populationSize;

			// read each line in the file and skip all the commented and empty line with the defaukt comment "#"
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

			bool linit = !(probdim == nDim && maxgens == maxGenerations && datanum == populationSize);
			if (linit)
			{
				if (!reset(nDim, maxGenerations, populationSize))
				{
					return false;
				}
			}

			f.rewindFile();

			// read each line in the file and skip all the commented and empty line with the defaukt comment "#"
			while (f.readLine())
			{
				// Parse the line into line arguments
				p.parse(f.getLine());

				if (p.at<std::string>(0) == "TolCOV")
				{
					TolCOV = p.at<T>(1);
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
				std::string strt("B" + std::to_string(n));

				while (f.readLine())
				{
					p.parse(f.getLine());

					if (p.at<std::string>(0) == strt)
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

			//! 0: uniform
			if (priorType == 0)
			{
				for (n = 0; n < nDim; n++)
				{
					priorParam1[n] = lowerBound[n];
					priorParam2[n] = upperBound[n];
				}
			}

			//! 1: gaussian
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

			//! 2: exponential
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

			//! 3: gamma
			if (priorType == 3)
			{
				f.rewindFile();

				while (f.readLine())
				{
					p.parse(f.getLine());

					//! \f$ \alpha \f$ parameter in Gamma distribution
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

					//! \f$ \beta \f$ parameter in Gamma distribution
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

			//! 4:composite
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
					std::string strt("C" + std::to_string(n));

					while (f.readLine())
					{
						p.parse(f.getLine());

						if (p.at<std::string>(0) == strt)
						{
							compositePriorDistribution[n] = p.at<int>(1);
							priorParam1[n] = p.at<T>(2);
							priorParam2[n] = p.at<T>(3);
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
		return false;
	}
	UMUQFAILRETURN("Requested File does not exist in the current PATH!!");
}

template <typename T>
bool stdata<T>::load(std::string const &fname)
{
	return load(&fname[0]);
}

#endif
