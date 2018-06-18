#ifndef UMUQ_STDATA_H
#define UMUQ_STDATA_H

#include "../core/core.hpp"
#include "../io/io.hpp"
#include "../misc/parser.hpp"

/*! \class optimizationParameters
 * \brief This is a class to set the optimization parameters 
 * 
 * \tparam T 
 */
template <typename T>
struct optimizationParameters
{
	int MaxIter;
	int Display;
	T Tolerance;
	T Step;

	//! constructor
	/*!
     *  \brief constructor for the default variables
     *    
     */
	optimizationParameters() : MaxIter(100),
							   Display(0),
							   Tolerance(1e-6),
							   Step(1e-5){};
};

/*! \file stdata.hpp
*  \brief stream Data type
*
* \param nDim
* \param maxGenerations
* \param populationSize
* \param lowerBound
* \param upperBound
* \param compositePriorDistribution
* \param priorMu
* \param priorSigma
* \param auxilSize
* \param auxilData
* \param minChainLength
* \param maxChainLength
* \param lb                  generic lower bound
* \param ub                  generic upper bound
* \param TolCOV
* \param bbeta
* \param seed
* \param options
* \param samplingType       sampling type which can be 0: uniform, 1: gaussian, 2: file 
* \param priorType          prior type which can be  0: lognormal, 1: gaussian
* \param priorCount
* \param iPlot
* \param saveData              dump current dataset of accepted points
* \param eachPopulationSize
* \param lastPopulationSize
* \param useCmaProposal
* \param initMean
* \param localCovariance
* \param useLocalCovariance
* \param localScale
*/
template <typename T>
class stdata
{
  public:
	//! constructor
	/*!
    * \brief constructor for the default variables
    *    
    */
	stdata() : nDim(0),
			   maxGenerations(0),
			   populationSize(0),
			   lowerBound(nullptr),
			   upperBound(nullptr),
			   compositePriorDistribution(nullptr),
			   priorMu(nullptr),
			   priorSigma(nullptr),
			   auxilSize(0),
			   auxilData(nullptr),
			   minChainLength(0),
			   maxChainLength(1e6),
			   lb(0), /* Default LB, same for all */
			   ub(0),
			   TolCOV(1.0),
			   bbeta(0.2),
			   seed(280675),
			   options(),
			   priorType(0),
			   priorCount(0),
			   iPlot(0),
			   saveData(1),
			   eachPopulationSize(nullptr),
			   lastPopulationSize(0),
			   useCmaProposal(0),
			   initMean(nullptr),
			   localCovariance(nullptr),
			   useLocalCovariance(0),
			   localScale(0){};

	//! constructor
	/*!
    *  \brief constructor for the default input variables
    *    
    */
	stdata(int probdim, int maxgens, int datanum) : nDim(probdim),
													maxGenerations(maxgens),
													populationSize(datanum),
													lowerBound(nullptr),
													upperBound(nullptr),
													compositePriorDistribution(nullptr),
													priorMu(nullptr),
													priorSigma(nullptr),
													auxilSize(0),
													auxilData(nullptr),
													minChainLength(0),
													maxChainLength(1e6),
													lb(-6), /* Default LB, same for all */
													ub(6),
													TolCOV(1.0),
													bbeta(0.2),
													seed(280675),
													options(),
													priorType(0),
													priorCount(0),
													iPlot(0),
													saveData(1),
													eachPopulationSize(nullptr),
													lastPopulationSize(datanum),
													useCmaProposal(0),
													initMean(nullptr),
													localCovariance(nullptr),
													useLocalCovariance(0),
													localScale(0)
	{
		try
		{
			lowerBound = new T[nDim];
			upperBound = new T[nDim];
			priorMu = new T[nDim];
		}
		catch (std::bad_alloc &e)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
		}

		{
			int e = nDim;
			while (e--)
			{
				lowerBound[e] = 0;
				upperBound[e] = 0;
				priorMu[e] = 0;
			}
		}

		try
		{
			priorSigma = new T[nDim * nDim];
		}
		catch (std::bad_alloc &e)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
		}

		for (int i = 0, k = 0; i < nDim; i++)
		{
			for (int j = 0; j < nDim; j++, k++)
			{
				if (i == j)
				{
					priorSigma[k] = 1.0;
				}
				else
				{
					priorSigma[k] = 0.0;
				}
			}
		}

		try
		{
			eachPopulationSize = new int[maxGenerations];
		}
		catch (std::bad_alloc &e)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
		}

		{
			int e = maxGenerations;
			while (e--)
			{
				eachPopulationSize[e] = populationSize;
			}
		}
		try
		{
			localCovariance = new T *[populationSize];
		}
		catch (std::bad_alloc &e)
		{
			std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
			std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
		}

		for (int i = 0; i < populationSize; i++)
		{
			try
			{
				localCovariance[i] = new T[nDim * nDim];
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
			}

			for (int j = 0, l = 0; j < nDim; j++)
			{
				for (int k = 0; k < nDim; k++, l++)
				{
					if (j == k)
					{
						localCovariance[i][l] = 1.0;
					}
					else
					{
						localCovariance[i][l] = 0.0;
					}
				}
			}
		}
	}

	/*!
    *  \brief load the input file fname
    *
    * \param fname              name of the input file
    * \return true on success
    */
	bool load(const char *fname = "tmcmc.par");

	//! destructor
	/*!
    *  \brief destructor 
    *    
    */
	~stdata()
	{
		destroy();
	}

	/*!
    *  \brief destroy created memory 
    *
    */
	void destroy();

  public:
	int nDim;
	int maxGenerations; /* = MAXGENS*/
	int populationSize; /* = DATANUM*/

	int *eachPopulationSize; /*[MAXGENS];*/
	int lastPopulationSize;

	T *lowerBound; /*[PROBDIM];*/
	T *upperBound; /*[PROBDIM];*/

	T *compositePriorDistribution; /*[PROBDIM]*/

	T *priorMu;
	T *priorSigma;

	int auxilSize;
	T *auxilData;

	int minChainLength;
	int maxChainLength;

	T lb; /*generic lower bound*/
	T ub; /*generic upper bound*/

	T TolCOV; /*a prescribed tolerance*/
	T bbeta;
	long seed;

	optimizationParameters<T> options;

	int samplingType; /* 0: uniform, 1: gaussian, 2: file */
	int priorType;	/* 0: lognormal, 1: gaussian */

	/* prior information needed for hiegherarchical analysis */
	/* this number = prior + number of datasets = 1 + N_IND */
	/* if it is 0, we only do the TMCMC */
	int priorCount;

	int iPlot;
	int saveData;

	int useCmaProposal;
	T **initMean; /* [DATANUM][PROBDIM] */

	T **localCovariance; /* [DATANUM][PROBDIM*PROBDIM] */
	int useLocalCovariance;
	T localScale;
};

// load the input file fname for setting the input variables
template<typename T>
bool stdata<T>::load(const char *fname)
{
	// We use an IO object to open and read a file
	io f;

	if (f.openFile(fname, f.in))
	{
		// We need a parser object to parse
		parser p;

		int probdim = nDim;
		int maxgens = maxGenerations;
		int datanum = populationSize;
		bool linit;

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
			else if (p.at<std::string>(0) == "TolCOV")
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
			else if (p.at<std::string>(0) == "priorCount")
			{
				priorCount = p.at<int>(1);
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

		linit = !(probdim == nDim && maxgens == maxGenerations && datanum == populationSize && lowerBound != nullptr);
		if (linit)
		{
			if (lowerBound != nullptr)
			{
				delete[] lowerBound;
			}

			try
			{
				lowerBound = new T[nDim];
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
				return false;
			}

			if (upperBound != nullptr)
			{
				delete[] upperBound;
			}

			try
			{
				upperBound = new T[nDim];
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
				return false;
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

			if (!found)
			{
				lowerBound[n] = lb; /* Bdef value or Default LB */
				upperBound[n] = ub; /* Bdef value of Default UB */
			}
		}

		if (priorType == 1) /* gaussian */
		{
			if (linit)
			{
				/* new, parse priorMu */
				if (priorMu != nullptr)
				{
					delete[] priorMu;
				}

				try
				{
					priorMu = new T[nDim];
				}
				catch (std::bad_alloc &e)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
					return false;
				}
			}

			f.rewindFile();
			found = 0;

			while (f.readLine())
			{
				p.parse(f.getLine());

				if (p.at<std::string>(0) == "priorMu")
				{
					for (n = 0; n < nDim; n++)
					{
						priorMu[n] = p.at<T>(n + 1);
					}
					found = 1;
					break;
				}
			}

			if (!found)
			{
				n = nDim;
				while (n--)
				{
					priorMu[n] = 0.0;
				}
			}

			if (linit)
			{
				/* new, parse priorSigma */
				if (priorSigma != nullptr)
				{
					delete[] priorSigma;
				}

				try
				{
					priorSigma = new T[nDim * nDim];
				}
				catch (std::bad_alloc &e)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
					return false;
				}
			}

			f.rewindFile();
			found = 0;

			while (f.readLine())
			{
				p.parse(f.getLine());

				if (p.at<std::string>(0) == "priorSigma")
				{
					for (n = 0; n < nDim * nDim; n++)
					{
						priorSigma[n] = p.at<T>(n + 1);
					}
					found = 1;
					break;
				}
			}

			if (!found)
			{
				int i, j, k;
				for (i = 0, k = 0; i < nDim; i++)
				{
					for (j = 0; j < nDim; j++, k++)
					{
						if (i == j)
						{
							priorSigma[k] = 1.0;
						}
						else
						{
							priorSigma[k] = 0.0;
						}
					}
				}
			}
		}

		if (priorType == 3) /* composite */
		{
			if (compositePriorDistribution != nullptr)
			{
				delete[] compositePriorDistribution;
			}
			try
			{
				compositePriorDistribution = new T[nDim];
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
				return false;
			}

			if (linit)
			{
				if (priorMu != nullptr)
				{
					delete[] priorMu;
				}

				try
				{
					priorMu = new T[nDim];
				}
				catch (std::bad_alloc &e)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
					return false;
				}

				if (priorSigma != nullptr)
				{
					delete[] priorSigma;
				}

				try
				{
					priorSigma = new T[nDim * nDim];
				}
				catch (std::bad_alloc &e)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
					return false;
				}
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
		found = 0;

		while (f.readLine())
		{
			p.parse(f.getLine());

			if (p.at<std::string>(0) == "auxilSize")
			{
				auxilSize = p.at<int>(1);
				found = 1;
				break;
			}
		}

		if (auxilSize > 0)
		{
			if (auxilData != nullptr)
			{
				delete[] auxilData;
			}

			try
			{
				auxilData = new T[auxilSize];
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
				return false;
			}

			f.rewindFile();

			found = 0;

			while (f.readLine())
			{
				p.parse(f.getLine());

				if (p.at<std::string>(0) == "auxilData")
				{
					for (n = 0; n < auxilSize; n++)
					{
						auxilData[n] = p.at<T>(n + 1);
					}
					found = 1;
					break;
				}
			}

			if (!found)
			{
				int i;
				for (i = 0; i < auxilSize; i++)
				{
					auxilData[i] = 0;
				}
			}
		}

		f.closeFile();

		if (linit)
		{
			if (eachPopulationSize != nullptr)
			{
				delete[] eachPopulationSize;
			}

			try
			{
				eachPopulationSize = new int[maxGenerations];
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
				return false;
			}

			n = maxGenerations;
			while (n--)
			{
				eachPopulationSize[n] = populationSize;
			}
			lastPopulationSize = populationSize;

			if (localCovariance != nullptr)
			{
				n = populationSize;
				while (n--)
				{
					if (localCovariance[n] != nullptr)
					{
						delete[] localCovariance[n];
					}
				}

				delete[] localCovariance;
			}

			try
			{
				localCovariance = new T *[populationSize];
			}
			catch (std::bad_alloc &e)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
				return false;
			}
			for (n = 0; n < populationSize; n++)
			{
				try
				{
					localCovariance[n] = new T[nDim * nDim];
				}
				catch (std::bad_alloc &e)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
					return false;
				}
				for (int i = 0, l = 0; i < nDim; i++)
				{
					for (int j = 0; j < nDim; j++, l++)
					{
						if (i == j)
						{
							localCovariance[n][l] = 1;
						}
						else
						{
							localCovariance[n][l] = 0;
						}
					}
				}
			}
		}
		return true;
	}
	return false;
}

/*!
*  \brief destroy the allocated memory 
*    
*/
template<typename T>
void stdata<T>::destroy()
{
	if (lowerBound != nullptr)
	{
		delete[] lowerBound;
		lowerBound = nullptr;
	}
	if (upperBound != nullptr)
	{
		delete[] upperBound;
		upperBound = nullptr;
	}
	if (compositePriorDistribution != nullptr)
	{
		delete[] compositePriorDistribution;
		compositePriorDistribution = nullptr;
	}
	if (priorMu != nullptr)
	{
		delete[] priorMu;
		priorMu = nullptr;
	}
	if (priorSigma != nullptr)
	{
		delete[] priorSigma;
		priorSigma = nullptr;
	}
	if (auxilData != nullptr)
	{
		delete[] auxilData;
		auxilData = nullptr;
	}
	if (eachPopulationSize != nullptr)
	{
		delete[] eachPopulationSize;
		eachPopulationSize = nullptr;
	}
	if (initMean != nullptr)
	{
		delete[] * initMean;
		delete[] initMean;
		initMean = nullptr;
	}
	if (localCovariance != nullptr)
	{
		delete[] * localCovariance;
		delete[] localCovariance;
		localCovariance = nullptr;
	}
}

#endif