#ifndef UMUQ_RUNINFO_H
#define UMUQ_RUNINFO_H

#include "../io/io.hpp"
#include "../misc/parser.hpp"

/*! \class runinfo
 * \ingroup data
 * 
 * \brief This class contains the run information of the TMCMC
 * 
 */
template <typename T>
class runinfo
{
  public:
	/*!
     * \brief constructor for the default variables
     *    
     */
	runinfo() : nDim(0),
				maxGenerations(0),
				Generation(0)
	{
	}

	/*!
	 * \brief Construct a new runinfo object
	 * 
	 * \param ProbDim          Dimension
	 * \param MaxGenerations   Max generations
	 */
	runinfo(int ProbDim, int MaxGenerations) : nDim(ProbDim),
											   maxGenerations(MaxGenerations),
											   Generation(0)
	{
		if (!init())
		{
			throw(std::runtime_error("Failed to initialiaze!"));
		}
	}

	/*!
	 * \brief Move constructor, construct a new runinfo object from an input runinfo object
	 * 
	 * \param inputRI Input runinfo object
	 */
	runinfo(runinfo<T> &&inputRI)
	{
		nDim = inputRI.nDim;
		maxGenerations = inputRI.maxGenerations;
		Generation = inputRI.Generation;
		CoefVar = std::move(inputRI.CoefVar);
		p = std::move(inputRI.p);
		currentuniques = std::move(inputRI.currentuniques);
		logselection = std::move(inputRI.logselection);
		acceptance = std::move(inputRI.acceptance);
		SS = std::move(inputRI.SS);
		meantheta = std::move(inputRI.meantheta);
	}

	/*!
	 * \brief Move assignment operator
	 * 
	 * \param inputRI      Input runinfo object
	 * \return runinfo<T>& 
	 */
	runinfo<T> &operator=(runinfo<T> &&inputRI)
	{
		nDim = inputRI.nDim;
		maxGenerations = inputRI.maxGenerations;
		Generation = inputRI.Generation;
		CoefVar = std::move(inputRI.CoefVar);
		p = std::move(inputRI.p);
		currentuniques = std::move(inputRI.currentuniques);
		logselection = std::move(inputRI.logselection);
		acceptance = std::move(inputRI.acceptance);
		SS = std::move(inputRI.SS);
		meantheta = std::move(inputRI.meantheta);

		return *this;
	}

	/*!
     * \brief destructor
     *    
     */
	~runinfo() {}

	/*!
     * \brief Initialize the database class register task
     *  
     * \returns false if there is not enough memory
     */
	bool init()
	{
		try
		{
			CoefVar.reset(new T[maxGenerations]());
			p.reset(new T[maxGenerations]());
			currentuniques.reset(new int[maxGenerations]());
			logselection.reset(new T[maxGenerations]());
			acceptance.reset(new T[maxGenerations]());
			SS.reset(new T[nDim * nDim]);
			meantheta.reset(new T[maxGenerations * nDim]);
		}
		catch (std::bad_alloc &e)
		{
			UMUQFAILRETURN("Failed to allocate memory!");
		}

		// set the first value to a high number
		CoefVar[0] = std::numeric_limits<T>::max();

		return true;
	}

	/*!
	 * \brief Reset the runinfo object with the new dimension and maximum generations
	 * 
	 * \param ProbDim         Dimension
	 * \param MaxGenerations  Maximum generations
	 * 
	 * \return true 
	 * \return false          if there is not enough memory
	 */
	bool reset(int ProbDim, int MaxGenerations)
	{
		nDim = ProbDim;
		maxGenerations = MaxGenerations;
		return init();
	}

	/*!
	 * \brief Exchanges the given runinfo object
	 * 
	 * \param inputRI Input runinfo object
	 */
	void swap(runinfo<T> &inputRI)
	{
		std::swap(nDim, inputRI.nDim);
		std::swap(maxGenerations, inputRI.maxGenerations);
		std::swap(Generation, inputRI.Generation);
		CoefVar.swap(inputRI.CoefVar);
		p.swap(inputRI.p);
		currentuniques.swap(inputRI.currentuniques);
		logselection.swap(inputRI.logselection);
		acceptance.swap(inputRI.acceptance);
		SS.swap(inputRI.SS);
		meantheta.swap(inputRI.meantheta);
	}

	/*!
     * \brief save the inofmration in a file @fileName
     * 
     * Write the runinfo data information to a file @fileName
     * 
     * \param fileName Name of the file (default name is runinfo.txt) for writing information 
     */
	bool save(const char *fileName = "runinfo.txt")
	{
		// Create an instance of the IO object
		io f;
		if (f.openFile(fileName, f.out | f.trunc))
		{
			// Get the IO stream
			std::fstream &fs = f.getFstream();
			bool tmp = true;

			fs << std::fixed;

			fs << "ProblemDimension= " << nDim << "\n";
			fs << "maxGenerations= " << maxGenerations << "\n";
			fs << "Generation= " << Generation << "\n";

			fs << "CoefVar[" << maxGenerations << "]=\n";
			tmp &= f.saveMatrix<T>(CoefVar, maxGenerations);

			fs << "p[" << maxGenerations << "]=\n";
			tmp &= f.saveMatrix<T>(p.get(), maxGenerations);

			fs << "currentuniques[" << maxGenerations << "]=\n";
			tmp &= f.saveMatrix<int>(currentuniques.get(), maxGenerations);

			fs << "logselection[" << maxGenerations << "]=\n";
			tmp &= f.saveMatrix<T>(logselection.get(), maxGenerations);

			fs << "acceptance[" << maxGenerations << "]=\n";
			tmp &= f.saveMatrix<T>(acceptance.get(), maxGenerations);

			fs << "SS[" << nDim << "][" << nDim << "]=\n";
			tmp &= f.saveMatrix<T>(SS.get(), nDim, nDim);

			fs << "meantheta[" << maxGenerations << "][" << nDim << "]=\n";
			tmp &= f.saveMatrix<T>(meantheta.get(), maxGenerations, nDim);

			f.closeFile();
			return tmp;
		}
		return false;
	}

	bool save(std::string const &fileName)
	{
		return save(&fileName[0]);
	}

	/*!
     * \brief load inofmration from a file @fileName
     * 
     * Load the runinfo data information from a file @fileName
     * 
     * \param fileName Name of the file (default name is runinfo.txt) for reading information 
     */
	bool load(const char *fileName = "runinfo.txt")
	{
		// Create an instance of the IO object
		io f;
		if (f.openFile(fileName, f.in))
		{
			// Create an instance of the parser object
			parser prs;
			bool tmp;

			tmp = f.readLine();

			prs.parse(f.getLine().substr(18));
			int ProbDim = prs.at<int>(0);

			tmp &= f.readLine();
			prs.parse(f.getLine().substr(16));
			int MaxGenerations = prs.at<int>(0);

			if (ProbDim != nDim || MaxGenerations != maxGenerations)
			{
				if (!reset(ProbDim, MaxGenerations))
				{
					return false;
				}
			}

			tmp &= f.readLine();
			prs.parse(f.getLine().substr(12));
			Generation = prs.at<int>(0);

			tmp &= f.readLine();
			tmp &= f.loadMatrix<T>(CoefVar.get(), maxGenerations);

			tmp &= f.readLine();
			tmp &= f.loadMatrix<T>(p.get(), maxGenerations);

			tmp &= f.readLine();
			tmp &= f.loadMatrix<int>(currentuniques.get(), maxGenerations);

			tmp &= f.readLine();
			tmp &= f.loadMatrix<T>(logselection.get(), maxGenerations);

			tmp &= f.readLine();
			tmp &= f.loadMatrix<T>(acceptance.get(), maxGenerations);

			tmp &= f.readLine();
			tmp &= f.loadMatrix<T>(SS.get(), nDim, nDim);

			tmp &= f.readLine();
			tmp &= f.loadMatrix<T>(meantheta.get(), maxGenerations, nDim);

			f.closeFile();
			return tmp;
		}
		return false;
	}

	bool load(std::string const &fileName)
	{
		return load(&fileName[0]);
	}

  private:
	// Make it noncopyable
	runinfo(runinfo<T> const &) = delete;

	// Make it not assignable
	runinfo<T> &operator=(runinfo<T> const &) = delete;

  public:
	//! Dimension
	int nDim;
	//! Max generations
	int maxGenerations;
	//! Current generation
	int Generation;

	//! The coefficient of variation of the plausibility weights [maxGenerations]
	std::unique_ptr<T[]> CoefVar;
	//! p cluster-wide                                           [maxGenerations]
	std::unique_ptr<T[]> p;
	//!                                                          [maxGenerations]
	std::unique_ptr<int[]> currentuniques;
	//!                                                          [maxGenerations]
	std::unique_ptr<T[]> logselection;
	//!                                                          [maxGenerations]
	std::unique_ptr<T[]> acceptance;
	//! SS cluster-wide                                          [nDim][nDim]
	std::unique_ptr<T[]> SS;
	//!                                                          [maxGenerations][nDim]
	std::unique_ptr<T[]> meantheta;
};

#endif //UMUQ_RUNINFO_H
