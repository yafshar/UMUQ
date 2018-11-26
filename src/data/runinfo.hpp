#ifndef UMUQ_RUNINFO_H
#define UMUQ_RUNINFO_H

#include "core/core.hpp"
#include "mpidatatype.hpp"
#include "misc/parser.hpp"
#include "io/io.hpp"

namespace umuq
{

namespace tmcmc
{

/*!
 * \ingroup TMCMC_Module
 * 
 * \brief Broadcasts running information to all processes of the group 
 * 
 * \tparam DataType Data type
 * 
 * \param other runinfo object which is casted to long long
 */
template <typename DataType>
void broadcastTask(long long const other);

//! True if update_Task has been registered, and false otherwise (logical).
template <typename DataType>
static bool isBroadcastTaskRegistered = false;

//! Mutex object for data broadcast
static std::mutex broadcastTask_m;

/*! \class runinfo
 * \ingroup TMCMC_Module
 * 
 * \brief This class contains the running information of the TMCMC algorithm
 * 
 */
template <typename DataType>
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
	runinfo(runinfo<DataType> &&other);

	/*!
     * \brief Move assignment operator
     * 
     * \param other      Input runinfo object
     * \return runinfo<DataType>& 
     */
	runinfo<DataType> &operator=(runinfo<DataType> &&other);

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
	void swap(runinfo<DataType> &other);

	/*!
     * \brief Save the information in a file fileName
     * Write the runinfo data information to a file fileName
     * 
     * \param fileName Name of the file (default name is runinfo.txt) for writing information
     */
	bool save(const char *fileName = "runinfo.txt");

	/*!
     * \brief Save the information in a file fileName
     * Write the runinfo data information to a file fileName
     * 
     * \param fileName Name of the file (default name is runinfo.txt) for writing information 
     */
	bool save(std::string const &fileName);

	/*!
	 * \brief Print the information on the standard output
	 * 
	 */
	void print();

	/*!
     * \brief Load information from a file fileName
     * Load the runinfo data information from a file fileName
     * 
     * \param fileName  Name of the file (default name is runinfo.txt) for reading information
     * 
     * \return true 
     * \return false 
     */
	bool load(const char *fileName = "runinfo.txt");

	/*!
     * \brief Load information from a file fileName
     * Load the runinfo data information from a file fileName
     * 
     * \param fileName  Name of the file (default name is runinfo.txt) for reading information
     * 
     * \return true 
     * \return false 
     */
	bool load(std::string const &fileName);

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
	inline bool setAcceptanceRate(DataType const acceptanceRate);

	/**
	 * \brief Broadcasts running information from to all the processes of the group
	 * 
	 */
	inline void broadcast();

	/*!
     * \brief Printing the running information mean & covariance computed from a collection 
	 * (the sample) of data.
	 * 
	 * Print the sample mean and the sample covariance
	 * 
     */
	void printSampleStatistics();

  private:
	/*!
     * \brief Delete a runinfo object copy construction
     * 
     * Make it noncopyable.
     */
	runinfo(runinfo<DataType> const &) = delete;

	/*!
     * \brief Delete a runinfo object assignment
     * 
     * Make it nonassignable
     * 
     * \returns runinfo<DataType>& 
     */
	runinfo<DataType> &operator=(runinfo<DataType> const &) = delete;

  public:
	//! Number of dimensions
	int nDim;
	//! Maximum number of generations
	int maxGenerations;
	//! Current generation number
	int currentGeneration;

	//! The coefficient of variation of the plausibility of the weights
	std::vector<DataType> CoefVar;
	//! Probabilty at each generation
	std::vector<DataType> generationProbabilty;
	//! Unique samples in the current generation
	std::vector<int> currentUniques;
	/*!
	 * In the Bayesian framework \f$ p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}, \f$ where 
	 * \f$ p(D) \f$ is the evidence and it can be estimated as a by-product in the TMCMC method. <br>
	 * logselection at each stage is used to estimate the log-evidence. <br>
	 * The log-evidence is estimated as a by-product of the method from 
	 * \f$ \ln p(D) \approx \ln{\prod_{j=1}^{m-1}\left( \frac{1}{N_j} \sum_{k=1}^{N_j} w_{j,k} \right )}. \f$ 
	 */
	std::vector<DataType> logselection;
	//! Each generation acceptance rate
	std::vector<DataType> acceptance;
	//! Samples covariance \f$ COV(\Theta(j)) \f$ at the generation \f$ j \f$
	std::vector<DataType> covariance;
	//! The sample mean \f$ \Theta(j) \f$ at the generation \f$ j \f$
	std::vector<DataType> meantheta;
};

template <typename DataType>
runinfo<DataType>::runinfo() : nDim(0),
							   maxGenerations(0),
							   currentGeneration(0)
{
	{
		std::lock_guard<std::mutex> lock(broadcastTask_m);
		if (!isBroadcastTaskRegistered<DataType>)
		{
			torc_register_task((void *)broadcastTask<DataType>);
			isBroadcastTaskRegistered<DataType> = true;
		}
	}
}

template <typename DataType>
runinfo<DataType>::runinfo(int ProbDim, int MaxGenerations) : nDim(ProbDim),
															  maxGenerations(MaxGenerations),
															  currentGeneration(0)
{
	if (!init())
	{
		UMUQFAIL("Failed to initialize!");
	}

	{
		std::lock_guard<std::mutex> lock(broadcastTask_m);
		if (!isBroadcastTaskRegistered<DataType>)
		{
			torc_register_task((void *)broadcastTask<DataType>);
			isBroadcastTaskRegistered<DataType> = true;
		}
	}
}

template <typename DataType>
runinfo<DataType>::runinfo(runinfo<DataType> &&other)
{
	nDim = other.nDim;
	maxGenerations = other.maxGenerations;
	currentGeneration = other.currentGeneration;
	CoefVar = std::move(other.CoefVar);
	generationProbabilty = std::move(other.generationProbabilty);
	currentUniques = std::move(other.currentUniques);
	logselection = std::move(other.logselection);
	acceptance = std::move(other.acceptance);
	covariance = std::move(other.covariance);
	meantheta = std::move(other.meantheta);
}

template <typename DataType>
runinfo<DataType> &runinfo<DataType>::operator=(runinfo<DataType> &&other)
{
	nDim = other.nDim;
	maxGenerations = other.maxGenerations;
	currentGeneration = other.currentGeneration;
	CoefVar = std::move(other.CoefVar);
	generationProbabilty = std::move(other.generationProbabilty);
	currentUniques = std::move(other.currentUniques);
	logselection = std::move(other.logselection);
	acceptance = std::move(other.acceptance);
	covariance = std::move(other.covariance);
	meantheta = std::move(other.meantheta);

	return *this;
}

template <typename DataType>
bool runinfo<DataType>::init()
{
	try
	{
		CoefVar.resize(maxGenerations, DataType{});
		generationProbabilty.resize(maxGenerations, DataType{});
		currentUniques.resize(maxGenerations, 0);
		logselection.resize(maxGenerations, DataType{});
		acceptance.resize(maxGenerations, DataType{});

		covariance.resize(nDim * nDim, DataType{});
		meantheta.resize(maxGenerations * nDim, DataType{});
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}

	// set the first value to a high number
	CoefVar[0] = std::numeric_limits<DataType>::max();

	return true;
}

template <typename DataType>
bool runinfo<DataType>::reset(int ProbDim, int MaxGenerations)
{
	nDim = ProbDim;
	maxGenerations = MaxGenerations;
	return init();
}

template <typename DataType>
void runinfo<DataType>::swap(runinfo<DataType> &other)
{
	std::swap(nDim, other.nDim);
	std::swap(maxGenerations, other.maxGenerations);
	std::swap(currentGeneration, other.currentGeneration);
	CoefVar.swap(other.CoefVar);
	generationProbabilty.swap(other.generationProbabilty);
	currentUniques.swap(other.currentUniques);
	logselection.swap(other.logselection);
	acceptance.swap(other.acceptance);
	covariance.swap(other.covariance);
	meantheta.swap(other.meantheta);
}

template <typename DataType>
bool runinfo<DataType>::save(const char *fileName)
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
		fs << "Generation= " << currentGeneration << "\n";

		fs << "CoefVar[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<DataType>(CoefVar.data(), maxGenerations);

		fs << "generationProbabilty[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<DataType>(generationProbabilty.data(), maxGenerations);

		fs << "currentUniques[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<int>(currentUniques.data(), maxGenerations);

		fs << "logselection[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<DataType>(logselection.data(), maxGenerations);

		fs << "acceptance[" << maxGenerations << "]=\n";
		tmp &= f.saveMatrix<DataType>(acceptance.data(), maxGenerations);

		fs << "covariance[" << nDim << "][" << nDim << "]=\n";
		tmp &= f.saveMatrix<DataType>(covariance.data(), nDim, nDim);

		fs << "meantheta[" << maxGenerations << "][" << nDim << "]=\n";
		tmp &= f.saveMatrix<DataType>(meantheta.data(), maxGenerations, nDim);

		f.closeFile();
		return tmp;
	}
	return false;
}

template <typename DataType>
bool runinfo<DataType>::save(std::string const &fileName)
{
	return save(&fileName[0]);
}

template <typename DataType>
void runinfo<DataType>::print()
{
	std::cout << "\n----------------------------\n"
			  << std::endl;
	std::cout << "Problem Dimension= " << nDim << "\n";
	std::cout << "Generation= " << currentGeneration << "\n";

	// Create an instance of the IO object
	umuq::io f;

	{
		// Define the printing format
		umuq::ioFormat umuqFormat = {"\n", "", "[\n", "]\n"};

		std::cout << "\nEach generation coefficient of variation=\n";
		f.setWidth(-1);
		f.printMatrix<DataType>(CoefVar, currentGeneration, 1, umuqFormat);

		std::cout << "\nEach generation probabilty=\n";
		f.setWidth(-1);
		f.printMatrix<DataType>(generationProbabilty, currentGeneration, 1, umuqFormat);

		std::cout << "\nEach generation number of unique sample points=\n";
		f.setWidth(-1);
		f.printMatrix<int>(currentUniques, currentGeneration, 1, umuqFormat);

		std::cout << "\nEach generation log selection for computing evidence=\n";
		f.setWidth(-1);
		f.printMatrix<DataType>(logselection, currentGeneration, 1, umuqFormat);

		std::cout << "\nThe logarithm of evidence is=[" << std::accumulate(logselection.data(), logselection.data() + currentGeneration, DataType{}) << "]\n";

		std::cout << "\nEach generation acceptance rate=\n";
		f.setWidth(-1);
		f.printMatrix<DataType>(acceptance, currentGeneration, 1, umuqFormat);
	}

	{
		// Define the printing format
		umuq::ioFormat umuqFormat = {" ", "", "[", "]\n"};

		std::cout << "\nEach generation mean of running information=\n";
		f.setWidth(-1);
		f.printMatrix<DataType>(meantheta, currentGeneration, nDim, umuqFormat);

		std::cout << "\nCovariance of running information at generation[" << currentGeneration << "]=\n";
		f.setWidth(-1);
		f.printMatrix<DataType>(covariance, nDim, nDim, umuqFormat);
	}

	std::cout << "\n----------------------------\n"
			  << std::endl;
}

template <typename DataType>
bool runinfo<DataType>::load(const char *fileName)
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
		currentGeneration = prs.at<int>(0);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<DataType>(CoefVar.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<DataType>(generationProbabilty.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<int>(currentUniques.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<DataType>(logselection.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<DataType>(acceptance.data(), maxGenerations);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<DataType>(covariance.data(), nDim, nDim);

		tmp &= f.readLine();
		tmp &= f.loadMatrix<DataType>(meantheta.data(), maxGenerations, nDim);

		f.closeFile();
		return tmp;
	}
	return false;
}

template <typename DataType>
bool runinfo<DataType>::load(std::string const &fileName)
{
	return load(&fileName[0]);
}

template <typename DataType>
inline bool runinfo<DataType>::setUniqueNumber(int const nUniques)
{
	if (currentGeneration < maxGenerations)
	{
		currentUniques[currentGeneration] = nUniques;
		return true;
	}
	UMUQFAILRETURN("Generation number is greater than the defined maximum number!");
}

template <typename DataType>
inline bool runinfo<DataType>::setAcceptanceRate(DataType const acceptanceRate)
{
	if (currentGeneration < maxGenerations)
	{
		acceptance[currentGeneration] = acceptanceRate;
		return true;
	}
	UMUQFAILRETURN("Generation number is greater than the defined maximum number!");
}

template <typename DataType>
inline void runinfo<DataType>::broadcast()
{
	if (torc_num_nodes() == 1)
	{
		return;
	}

#if HAVE_MPI == 1
	for (int i = 0; i < torc_num_nodes(); i++)
	{
		torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())broadcastTask<DataType>, 1,
					   1, MPIDatatype<long long>, CALL_BY_REF,
					   reinterpret_cast<long long>(this));
	}
	torc_waitall();
#endif // MPI
}

template <typename DataType>
void runinfo<DataType>::printSampleStatistics()
{
	std::cout << "----------------------------" << std::endl;
	umuq::io f;
	// Define the printing format
	umuq::ioFormat meanFormat = {" ", "", "Mean=[", "]\nSample covariance matrix=\n"};
	umuq::ioFormat covarianceFormat = {" ", "\n", "[", "]"};
	f.printMatrix<DataType>(meantheta.data() + currentGeneration * nDim, 1, nDim, meanFormat);
	f.printMatrix<DataType>(covariance, nDim, nDim, covarianceFormat);
	std::cout << "----------------------------" << std::endl;
}

template <typename DataType>
void broadcastTask(long long const other)
{
#if HAVE_MPI == 1
	auto obj = reinterpret_cast<runinfo<DataType> *>(other);

	int const nDim(obj->nDim * obj->nDim);
	int const maxGenerations(obj->maxGenerations);

	MPI_Request request[3];

	MPI_Ibcast(obj->covariance.data(), nDim, MPIDatatype<DataType>, 0, MPI_COMM_WORLD, &request[0]);
	MPI_Ibcast(obj->generationProbabilty.data(), maxGenerations, MPIDatatype<DataType>, 0, MPI_COMM_WORLD, &request[1]);
	MPI_Ibcast(&obj->currentGeneration, 1, MPI_INT, 0, MPI_COMM_WORLD, &request[2]);

	MPI_Waitall(3, request, MPI_STATUSES_IGNORE);
#endif // MPI
}

} // namespace tmcmc
} // namespace umuq

#endif //UMUQ_RUNINFO
