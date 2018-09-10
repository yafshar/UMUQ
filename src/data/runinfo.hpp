#ifndef UMUQ_RUNINFO_H
#define UMUQ_RUNINFO_H

#include "misc/parser.hpp"
#include "io/io.hpp"

namespace umuq
{
/*! \namespace tmcmc
 * \brief Namespace containing all the functions for TMCMC algorithm
 *
 */
namespace tmcmc
{

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
	runinfo();

	/*!
     * \brief Construct a new runinfo object
     * 
     * \param ProbDim          Dimension
     * \param MaxGenerations   Max generations
     * 
     */
	runinfo(int ProbDim, int MaxGenerations);

	/*!
     * \brief Move constructor, construct a new runinfo object from an input runinfo object
     * 
     * \param other Input runinfo object
     * 
     */
	runinfo(runinfo<T> &&other);

	/*!
     * \brief Move assignment operator
     * 
     * \param other      Input runinfo object
     * \return runinfo<T>& 
     */
	runinfo<T> &operator=(runinfo<T> &&other);

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
	bool init();

	/*!
     * \brief Reset the runinfo object with the new dimension and maximum generations
     * 
     * \param ProbDim         Dimension
     * \param MaxGenerations  Maximum generations
     * 
     * \return true 
     * \return false          if there is not enough memory
     */
	bool reset(int ProbDim, int MaxGenerations);

	/*!
     * \brief Exchanges the given runinfo object
     * 
     * \param other Input runinfo object
     */
	void swap(runinfo<T> &other);

	/*!
     * \brief Save the inofmration in a file @fileName
     * Write the runinfo data information to a file @fileName
     * 
     * \param fileName Name of the file (default name is runinfo.txt) for writing information
     *  
     */
	bool save(const char *fileName = "runinfo.txt");

	bool save(std::string const &fileName);

	/*!
     * \brief Load inofmration from a file @fileName
     * Load the runinfo data information from a file @fileName
     * 
     * \param fileName  Name of the file (default name is runinfo.txt) for reading information
     * 
     * \return true 
     * \return false 
     */
	bool load(const char *fileName = "runinfo.txt");

	bool load(std::string const &fileName);

	/*!
	 * \brief Get the Uniques object
	 * Find the uniue nCols dimensions sample points in an array of nRows * nCols data
	 * 
	 * \param iArray    Input data
	 * \param nRows     Number of rows
	 * \param nCols     Number of columns (data dimension)
	 * \param uAarray   Unique data (every row in this data is unique)
	 */
	void getUniques(T const *iArray, int const nRows, int const nCols, std::vector<T> &uAarray);
	void getUniques(std::vector<T> const &iArray, int const nRows, int const nCols, std::vector<T> &uAarray);
	void getUniques(std::unique_ptr<T[]> const &iArray, int const nRows, int const nCols, std::vector<T> &uAarray);

	/*!
	 * \brief Set the number of uniques for the current generation
	 * 
	 * \param nUniques Number of unique samples
	 * 
	 * \return true 
	 * \return false If the current generation is greater than the defined maximum number 
	 */
	inline bool setUniqueNumber(int const nUniques);

	/*!
	 * \brief Set the Acceptance rate for the current generation
	 * 
	 * \param acceptanceRate 
	 * 
	 * \return true 
	 * \return false If the current generation is greater than the defined maximum number 
	 */
	inline bool setAcceptanceRate(T const acceptanceRate);

  private:
	// Make it noncopyable
	runinfo(runinfo<T> const &) = delete;

	// Make it nonassignable
	runinfo<T> &operator=(runinfo<T> const &) = delete;

  public:
	//! Dimension
	int nDim;
	//! Max generations
	int maxGenerations;
	//! Current generation
	int Generation;

	//! The coefficient of variation of the plausibility weights [maxGenerations]
	std::vector<T> CoefVar;
	//! p cluster-wide                                           [maxGenerations]
	//! probabilty at each generation
	std::vector<T> p;
	//!                                                          [maxGenerations]
	std::vector<int> currentuniques;
	//!                                                          [maxGenerations]
	std::vector<T> logselection;
	//!                                                          [maxGenerations]
	std::vector<T> acceptance;
	//! SS cluster-wide                                          [nDim][nDim]
	//!
	std::vector<T> SS;
	//!                                                          [maxGenerations][nDim]
	std::vector<T> meantheta;
};

/*!
 * \brief Construct a new runinfo object
 * 
 * \tparam T 
 */
template <typename T>
runinfo<T>::runinfo() : nDim(0),
						maxGenerations(0),
						Generation(0)
{
}

/*!
 * \brief Construct a new runinfo object
 * 
 * 
 * \param ProbDim         Dimension
 * \param MaxGenerations  Max generations
 */
template <typename T>
runinfo<T>::runinfo(int ProbDim, int MaxGenerations) : nDim(ProbDim),
													   maxGenerations(MaxGenerations),
													   Generation(0)
{
	if (!init())
	{
		UMUQFAIL("Failed to initialiaze!");
	}
}

/*!
 * \brief Move constructor, construct a new runinfo object from an input runinfo object
 * 
 * \param other Input runinfo object
 * 
 */
template <typename T>
runinfo<T>::runinfo(runinfo<T> &&other)
{
	nDim = other.nDim;
	maxGenerations = other.maxGenerations;
	Generation = other.Generation;
	CoefVar = std::move(other.CoefVar);
	p = std::move(other.p);
	currentuniques = std::move(other.currentuniques);
	logselection = std::move(other.logselection);
	acceptance = std::move(other.acceptance);
	SS = std::move(other.SS);
	meantheta = std::move(other.meantheta);
}

/*!
 * \brief Move assignment operator
 * 
 * \param other      Input runinfo object
 * \return runinfo<T>& 
 */
template <typename T>
runinfo<T> &runinfo<T>::operator=(runinfo<T> &&other)
{
	nDim = other.nDim;
	maxGenerations = other.maxGenerations;
	Generation = other.Generation;
	CoefVar = std::move(other.CoefVar);
	p = std::move(other.p);
	currentuniques = std::move(other.currentuniques);
	logselection = std::move(other.logselection);
	acceptance = std::move(other.acceptance);
	SS = std::move(other.SS);
	meantheta = std::move(other.meantheta);

	return *this;
}

/*!
 * \brief Initialize the database class register task
 *  
 * \returns false if there is not enough memory
 */
template <typename T>
bool runinfo<T>::init()
{
	try
	{
		CoefVar.resize(maxGenerations, T{});
		p.resize(maxGenerations, T{});
		currentuniques.resize(maxGenerations, 0);
		logselection.resize(maxGenerations, T{});
		acceptance.resize(maxGenerations, T{});

		SS.resize(nDim * nDim, T{});
		meantheta.resize(maxGenerations * nDim, T{});
	}
	catch (...)
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
 * \return false  If there is not enough memory
 */
template <typename T>
bool runinfo<T>::reset(int ProbDim, int MaxGenerations)
{
	nDim = ProbDim;
	maxGenerations = MaxGenerations;
	return init();
}

/*!
 * \brief Exchanges the given runinfo object
 * 
 * \param other Input runinfo object
 * 
 */
template <typename T>
void runinfo<T>::swap(runinfo<T> &other)
{
	std::swap(nDim, other.nDim);
	std::swap(maxGenerations, other.maxGenerations);
	std::swap(Generation, other.Generation);
	CoefVar.swap(other.CoefVar);
	p.swap(other.p);
	currentuniques.swap(other.currentuniques);
	logselection.swap(other.logselection);
	acceptance.swap(other.acceptance);
	SS.swap(other.SS);
	meantheta.swap(other.meantheta);
}

/*!
 * \brief Save the inofmration in a file @fileName
 * Write the runinfo data information to a file @fileName
 * 
 * \param fileName Name of the file (default name is runinfo.txt) for writing information 
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool runinfo<T>::save(const char *fileName)
{
	// Create an instance of the IO object
	umuq::io f;
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
		tmp &= f.saveMatrix<T>(CoefVar.data(), maxGenerations);

		fs << "p[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<T>(p.data(), maxGenerations);

		fs << "currentuniques[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<int>(currentuniques.data(), maxGenerations);

		fs << "logselection[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<T>(logselection.data(), maxGenerations);

		fs << "acceptance[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<T>(acceptance.data(), maxGenerations);

		fs << "SS[" << nDim << "][" << nDim << "]=\n";
		tmp &= f.saveMatrix<T>(SS.data(), nDim, nDim);

		fs << "meantheta[" << maxGenerations << "][" << nDim << "]=\n";
		tmp &= f.saveMatrix<T>(meantheta.data(), maxGenerations, nDim);

		f.closeFile();
		return tmp;
	}
	return false;
}

template <typename T>
bool runinfo<T>::save(std::string const &fileName)
{
	return save(&fileName[0]);
}

/*!
 * \brief Load inofmration from a file @fileName
 * Load the runinfo data information from a file @fileName
 * 
 * \param fileName  Name of the file (default name is runinfo.txt) for reading information
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool runinfo<T>::load(const char *fileName)
{
	// Create an instance of the IO object
	umuq::io f;
	if (f.openFile(fileName, f.in))
	{
		// Create an instance of the parser object
		umuq::parser prs;
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
		tmp &= f.loadMatrix<T>(CoefVar.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<T>(p.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<int>(currentuniques.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<T>(logselection.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<T>(acceptance.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<T>(SS.data(), nDim, nDim);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<T>(meantheta.data(), maxGenerations, nDim);

		f.closeFile();
		return tmp;
	}
	return false;
}

template <typename T>
bool runinfo<T>::load(std::string const &fileName)
{
	return load(&fileName[0]);
}

template <typename T>
void runinfo<T>::getUniques(T const *iArray, int const nRows, int const nCols, std::vector<T> &uArray)
{
	//! Resize the unique array to the maximum size
	uArray.resize(nRows * nCols);

	//! Create a temporary array with the size of number of columns (one row of data)
	std::vector<T> x(nCols);

	//! First element in the input array is considered unique
	std::copy(iArray, iArray + nCols, uArray.begin());

	//! We have one unique
	int nUniques(1);

	for (int i = 1; i < nRows; i++)
	{
		int const s = i * nCols;
		std::copy(iArray + s, iArray + s + nCols, x.begin());

		//! Consider this x rows is unique among all the rows
		bool uniqueFlag = true;

		//! check it with all the unique rows
		for (int j = 0, l = 0; j < nUniques; j++, l += nCols)
		{
			//! Consider they are the same
			bool compareFlag = true;
			for (int k = 0; k < nCols; k++)
			{
				if (std::abs(x[k] - uArray[l + k]) > 1e-6)
				{
					//! one element in the row differs, so they are different
					compareFlag = false;
					break;
				}
			}
			if (compareFlag)
			{
				//! It is not a unique row
				uniqueFlag = false;
				break;
			}
		}

		if (uniqueFlag)
		{
			int const e = nUniques * nCols;
			std::copy(x.begin(), x.end(), uArray.begin() + e);
			nUniques++;
		}
	}

	//! Correct the size of the unique array
	if (nUniques * nCols < uArray.size())
	{
		uArray.resize(nUniques * nCols);
	}
	return;
}

template <typename T>
void runinfo<T>::getUniques(std::vector<T> const &iArray, int const nRows, int const nCols, std::vector<T> &uArray)
{
	getUniques(iArray.data(), nRows, nCols, uArray);
}

template <typename T>
void runinfo<T>::getUniques(std::unique_ptr<T[]> const &iArray, int const nRows, int const nCols, std::vector<T> &uArray)
{
	getUniques(iArray.get(), nRows, nCols, uArray);
}

template <typename T>
inline bool runinfo<T>::setUniqueNumber(int const nUniques)
{
	if (Generation < maxGenerations)
	{
		currentuniques[Generation] = nUniques;
		return true;
	}
	UMUQFAILRETURN("Generation is greater than the defined maximum number!");
}

template <typename T>
inline bool runinfo<T>::setAcceptanceRate(T const acceptanceRate)
{
	if (Generation < maxGenerations)
	{
		acceptance[Generation] = acceptanceRate;
		return true;
	}
	UMUQFAILRETURN("Generation is greater than the defined maximum number!");
}

} // namespace tmcmc
} // namespace umuq

#endif //UMUQ_RUNINFO
