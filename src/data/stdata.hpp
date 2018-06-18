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
 * \param priorType                  Prior type which is :   0: lognormal, 1: gaussian
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
 * \param priorMu                    Prior mean
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
	stdata() : nDim(0),
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

	/*!
     * \brief constructor for the default input variables
     *    
     */
	stdata(int probdim, int MaxGenerations, int PopulationSize) : nDim(probdim),
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
																  options()
	{
		try
		{
			eachPopulationSize.reset(new int[maxGenerations]);
			lowerBound.reset(new T[nDim]());
			upperBound.reset(new T[nDim]());
			priorMu.reset(new T[nDim]());
			priorSigma.reset(new T[nDim * nDim]());
			localCovariance.reset(new T[populationSize * nDim * nDim]());
		}
		catch (std::bad_alloc &e)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
			throw(std::runtime_error("Failed to allocate memory!"));
		}

		for (int i = 0, k = 0; i < nDim; i++)
		{
			for (int j = 0; j < nDim; j++, k++)
			{
				if (i == j)
				{
					priorSigma[k] = static_cast<T>(1);
				}
			}
		}

		std::fill(eachPopulationSize.get(), eachPopulationSize.get() + maxGenerations, populationSize);

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

	/*!
	 * \brief Move constructor, construct a new stdata object from an input object
     * 
     * \param inputSD  Input stdata object
	 */
	stdata(stdata<T> &&inputSD)
	{
		nDim = inputSD.nDim;
		maxGenerations = inputSD.maxGenerations;
		populationSize = inputSD.populationSize;
		lastPopulationSize = inputSD.lastPopulationSize;
		auxilSize = inputSD.auxilData;
		minChainLength = inputSD.minChainLength;
		maxChainLength = inputSD.maxChainLength;
		seed = inputSD.seed;
		samplingType = inputSD.samplingType;
		priorType = inputSD.priorType;
		iPlot = inputSD.iPlot;
		saveData = inputSD.saveData;
		useCmaProposal = inputSD.useCmaProposal;
		useLocalCovariance = inputSD.useLocalCovariance;
		lb = inputSD.lb;
		ub = inputSD.ub;
		TolCOV = inputSD.TolCOV;
		bbeta = inputSD.bbeta;
		localScale = inputSD.localScale;
		options.Display = inputSD.options.Display;
		options.MaxIter = inputSD.options.MaxIter;
		options.Step = inputSD.options.Step;
		options.Tolerance = inputSD.options.Tolerance;
		eachPopulationSize = std::move(inputSD.eachPopulationSize);
		lowerBound = std::move(inputSD.lowerBound);
		upperBound = std::move(inputSD.upperBound);
		compositePriorDistribution = std::move(inputSD.compositePriorDistribution);
		priorMu = std::move(inputSD.priorMu);
		priorSigma = std::move(inputSD.priorSigma);
		auxilData = std::move(inputSD.auxilData);
		initMean = std::move(inputSD.initMean);
		localCovariance = std::move(inputSD.localCovariance);
	}

	/*!
	 * \brief Move assignment operator
	 * 
	 * \param inputSD 
	 * \return stdata<T>& 
	 */
	stdata<T> &operator=(stdata<T> &&inputSD)
	{
		nDim = inputSD.nDim;
		maxGenerations = inputSD.maxGenerations;
		populationSize = inputSD.populationSize;
		lastPopulationSize = inputSD.lastPopulationSize;
		auxilSize = inputSD.auxilData;
		minChainLength = inputSD.minChainLength;
		maxChainLength = inputSD.maxChainLength;
		seed = inputSD.seed;
		samplingType = inputSD.samplingType;
		priorType = inputSD.priorType;
		iPlot = inputSD.iPlot;
		saveData = inputSD.saveData;
		useCmaProposal = inputSD.useCmaProposal;
		useLocalCovariance = inputSD.useLocalCovariance;
		lb = inputSD.lb;
		ub = inputSD.ub;
		TolCOV = inputSD.TolCOV;
		bbeta = inputSD.bbeta;
		localScale = inputSD.localScale;
		options.Display = inputSD.options.Display;
		options.MaxIter = inputSD.options.MaxIter;
		options.Step = inputSD.options.Step;
		options.Tolerance = inputSD.options.Tolerance;
		eachPopulationSize = std::move(inputSD.eachPopulationSize);
		lowerBound = std::move(inputSD.lowerBound);
		upperBound = std::move(inputSD.upperBound);
		compositePriorDistribution = std::move(inputSD.compositePriorDistribution);
		priorMu = std::move(inputSD.priorMu);
		priorSigma = std::move(inputSD.priorSigma);
		auxilData = std::move(inputSD.auxilData);
		initMean = std::move(inputSD.initMean);
		localCovariance = std::move(inputSD.localCovariance);

		return *this;
	}

	/*!
	 * \brief Swap the stdata objects
	 * 
	 * \param inputSD 
	 */
	void swap(stdata<T> &inputSD)
	{
		std::swap(nDim, inputSD.nDim);
		std::swap(maxGenerations, inputSD.maxGenerations);
		std::swap(populationSize, inputSD.populationSize);
		std::swap(lastPopulationSize, inputSD.lastPopulationSize);
		std::swap(auxilSize, inputSD.auxilData);
		std::swap(minChainLength, inputSD.minChainLength);
		std::swap(maxChainLength, inputSD.maxChainLength);
		std::swap(seed, inputSD.seed);
		std::swap(samplingType, inputSD.samplingType);
		std::swap(priorType, inputSD.priorType);
		std::swap(iPlot, inputSD.iPlot);
		std::swap(saveData, inputSD.saveData);
		std::swap(useCmaProposal, inputSD.useCmaProposal);
		std::swap(useLocalCovariance, inputSD.useLocalCovariance);
		std::swap(lb, inputSD.lb);
		std::swap(ub, inputSD.ub);
		std::swap(TolCOV, inputSD.TolCOV);
		std::swap(bbeta, inputSD.bbeta);
		std::swap(localScale, inputSD.localScale);
		std::swap(options.Display, inputSD.options.Display);
		std::swap(options.MaxIter, inputSD.options.MaxIter);
		std::swap(options.Step, inputSD.options.Step);
		std::swap(options.Tolerance, inputSD.options.Tolerance);
		eachPopulationSize.swap(inputSD.eachPopulationSize);
		lowerBound.swap(inputSD.lowerBound);
		upperBound.swap(inputSD.upperBound);
		compositePriorDistribution.swap(inputSD.compositePriorDistribution);
		priorMu.swap(inputSD.priorMu);
		priorSigma.swap(inputSD.priorSigma);
		auxilData.swap(inputSD.auxilData);
		initMean.swap(inputSD.initMean);
		localCovariance.swap(inputSD.localCovariance);
	}

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
	bool reset(int probdim, int MaxGenerations, int PopulationSize)
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

			eachPopulationSize.reset();
			lowerBound.reset();
			upperBound.reset();
			priorMu.reset();
			priorSigma.reset();
			localCovariance.reset();

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
			eachPopulationSize.reset(new int[maxGenerations]);
			lowerBound.reset(new T[nDim]());
			upperBound.reset(new T[nDim]());
			priorMu.reset(new T[nDim]());
			priorSigma.reset(new T[nDim * nDim]());
			localCovariance.reset(new T[populationSize * nDim * nDim]());
		}
		catch (std::bad_alloc &e)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
			return false;
		}

		for (int i = 0, k = 0; i < nDim; i++)
		{
			for (int j = 0; j < nDim; j++, k++)
			{
				if (i == j)
				{
					priorSigma[k] = static_cast<T>(1);
				}
			}
		}

		std::fill(eachPopulationSize.get(), eachPopulationSize.get() + maxGenerations, populationSize);

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
     * \brief load the input file fname
     *
     * \param fname              name of the input file
     * \return true on success
     */
	bool load(const char *fname = "tmcmc.par");

	/*!
     * \brief Default destructor 
     *    
     */
	~stdata() {}

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

	//! Prior type which is :   0: lognormal, 1: gaussian
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
	std::unique_ptr<int[]> eachPopulationSize;

	//! Sampling domain lower bounds for each dimension
	std::unique_ptr<T[]> lowerBound;

	//! Sampling domain upper bounds for each dimension
	std::unique_ptr<T[]> upperBound;

	//! Composite distribution as a prior
	std::unique_ptr<T[]> compositePriorDistribution;

	//! Prior mean
	std::unique_ptr<T[]> priorMu;

	//! Prior standard deviation
	std::unique_ptr<T[]> priorSigma;

	//! Auxillary data
	std::unique_ptr<T[]> auxilData;

	//! Initial Mean with the size of [populationSize*nDim]
	std::unique_ptr<T[]> initMean;

	//! Local covariance with the size of [populationSize*nDim*nDim]
	std::unique_ptr<T[]> localCovariance;
};

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
	io f;

	if (f.openFile(fname, f.in))
	{
		// We need a parser object to parse
		parser p;

		//! These are temporary variables
		int probdim = nDim;
		int maxgens = maxGenerations;
		int datanum = populationSize;

		//read each line in the file and skip all the commented and empty line with the defaukt comment "#"
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
		
		//read each line in the file and skip all the commented and empty line with the defaukt comment "#"
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

			//In case we do not find the value, we use the default lower bound and upper bound
			if (!found)
			{
				lowerBound[n] = lb;
				upperBound[n] = ub;
			}
		}

		if (priorType == 1) /* gaussian */
		{
			f.rewindFile();

			while (f.readLine())
			{
				p.parse(f.getLine());

				if (p.at<std::string>(0) == "priorMu")
				{
					for (n = 0; n < nDim; n++)
					{
						priorMu[n] = p.at<T>(n + 1);
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
						priorSigma[n] = p.at<T>(n + 1);
					}
					break;
				}
			}
		}

		// Composite prior at input
		if (priorType == 3)
		{
			try
			{
				compositePriorDistribution.reset(new T[nDim]());
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
				return false;
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
						compositePriorDistribution[n] = p.at<T>(1);
						lowerBound[n] = p.at<T>(2);
						upperBound[n] = p.at<T>(3);
						found = 1;
						break;
					}
				}

				if (!found)
				{
					compositePriorDistribution[n] = 0;
					lowerBound[n] = lb; /* Bdef value or Default LB */
					upperBound[n] = ub; /* Bdef value of Default UB */
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
				auxilData.reset(new T[auxilSize]());
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
				return false;
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

#endif