#ifndef UMUQ_RUNINFO_H
#define UMUQ_RUNINFO_H

#include "core/core.hpp"
#include "datatype/mpidatatype.hpp"
#include "misc/parser.hpp"
#include "io/io.hpp"

#include <mutex>
#include <string>
#include <vector>
#include <utility>
#include <limits>
#include <fstream>
#include <numeric>

namespace umuq
{

namespace tmcmc
{

/*!
 * \ingroup TMCMC_Module
 *
 * \brief Broadcasts running information to all processes of the group
 *
 * \param other runinfo object which is casted to long long
 */
void broadcastTask(long long const other);

/*! True if update_Task has been registered, and false otherwise (logical). */
static bool isBroadcastTaskRegistered = false;

/*! Mutex object for data broadcast */
static std::mutex broadcastTask_m;

/*! \class runinfo
 * \ingroup TMCMC_Module
 *
 * \brief This class contains the running information of the TMCMC algorithm
 */
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
     */
    runinfo(int ProbDim, int MaxGenerations);

    /*!
     * \brief Move constructor, construct a new runinfo object from an input runinfo object
     *
     * \param other Input runinfo object
     */
    runinfo(runinfo &&other);

    /*!
     * \brief Move assignment operator
     *
     * \param other      Input runinfo object
     * \return runinfo&
     */
    runinfo &operator=(runinfo &&other);

    /*!
     * \brief destructor
     */
    ~runinfo();

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
     * \return false If there is not enough memory
     */
    bool reset(int ProbDim, int MaxGenerations);

    /*!
     * \brief Exchanges the given runinfo object
     *
     * \param other Input runinfo object
     */
    void swap(runinfo &other);

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
     */
    void print();

    /*!
     * \brief Load information from a file fileName
     * Load the runinfo data information from a file fileName
     *
     * \param fileName  Name of the file (default name is runinfo.txt) for reading information
     *
     * \return false If it encounters any problem
     */
    bool load(const char *fileName = "runinfo.txt");

    /*!
     * \brief Load information from a file fileName
     * Load the runinfo data information from a file fileName
     *
     * \param fileName  Name of the file (default name is runinfo.txt) for reading information
     *
     * \return false If it encounters any problem
     */
    bool load(std::string const &fileName);

    /*!
     * \brief Set the number of uniques for the current generation
     *
     * \param nUniques Number of unique samples
     *
     * \return false If the current generation is greater than the defined maximum number
     */
    inline bool setUniqueNumber(int const nUniques);

    /*!
     * \brief Set the Acceptance rate for the current generation
     *
     * \param acceptanceRate
     *
     * \return false If the current generation is greater than the defined maximum number
     */
    inline bool setAcceptanceRate(double const acceptanceRate);

    /*!
     * \brief Broadcasts running information from to all the processes of the group
     */
    inline void broadcast();

    /*!
     * \brief Printing the running information mean & covariance computed from a collection
     * (the sample) of data.
     *
     * Print the sample mean and the sample covariance
     */
    void printSampleStatistics();

private:
    /*!
     * \brief Delete a runinfo object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    runinfo(runinfo const &) = delete;

    /*!
     * \brief Delete a runinfo object assignment
     *
     * Avoiding implicit copy assignment.
     *
     * \returns runinfo&
     */
    runinfo &operator=(runinfo const &) = delete;

public:
    //! Number of dimensions
    int nDim;
    //! Maximum number of generations
    int maxGenerations;
    //! Current generation number
    int currentGeneration;

    //! The coefficient of variation of the plausibility of the weights
    std::vector<double> CoefVar;
    //! Probabilty at each generation
    std::vector<double> generationProbabilty;
    //! Unique samples in the current generation
    std::vector<int> currentUniques;

    /*!
     * In the Bayesian framework \f$ p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}, \f$ where
     * \f$ p(D) \f$ is the evidence and it can be estimated as a by-product in the TMCMC method. <br>
     * logselection at each stage is used to estimate the log-evidence. <br>
     * The log-evidence is estimated as a by-product of the method from
     * \f$ \ln p(D) \approx \ln{\prod_{j=1}^{m-1}\left( \frac{1}{N_j} \sum_{k=1}^{N_j} w_{j,k} \right )}. \f$
     */
    std::vector<double> logselection;
    //! Each generation acceptance rate
    std::vector<double> acceptance;
    //! Samples covariance \f$ COV(\Theta(j)) \f$ at the generation \f$ j \f$
    std::vector<double> covariance;
    //! The sample mean \f$ \Theta(j) \f$ at the generation \f$ j \f$
    std::vector<double> meantheta;
};

runinfo::runinfo() : nDim(0),
                     maxGenerations(0),
                     currentGeneration(0)
{
    {
        std::lock_guard<std::mutex> lock(broadcastTask_m);
        if (!isBroadcastTaskRegistered)
        {
            torc_register_task((void *)broadcastTask);
            isBroadcastTaskRegistered = true;
        }
    }
}

runinfo::runinfo(int ProbDim, int MaxGenerations) : nDim(ProbDim),
                                                    maxGenerations(MaxGenerations),
                                                    currentGeneration(0)
{
    if (!init())
    {
        UMUQFAIL("Failed to initialize!");
    }

    {
        std::lock_guard<std::mutex> lock(broadcastTask_m);
        if (!isBroadcastTaskRegistered)
        {
            torc_register_task((void *)broadcastTask);
            isBroadcastTaskRegistered = true;
        }
    }
}

runinfo::runinfo(runinfo &&other)
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

runinfo &runinfo::operator=(runinfo &&other)
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

runinfo::~runinfo() {}

bool runinfo::init()
{
    try
    {
        CoefVar.resize(maxGenerations, double{});
        generationProbabilty.resize(maxGenerations, double{});
        currentUniques.resize(maxGenerations, 0);
        logselection.resize(maxGenerations, double{});
        acceptance.resize(maxGenerations, double{});
        covariance.resize(nDim * nDim, double{});
        meantheta.resize(maxGenerations * nDim, double{});
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }

    // set the first value to a high number
    CoefVar[0] = std::numeric_limits<double>::max();
    return true;
}

bool runinfo::reset(int ProbDim, int MaxGenerations)
{
    nDim = ProbDim;
    maxGenerations = MaxGenerations;
    return init();
}

void runinfo::swap(runinfo &other)
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

bool runinfo::save(const char *fileName)
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
        tmp &= f.saveMatrix(CoefVar.data(), maxGenerations);

        fs << "generationProbabilty[" << maxGenerations << "]=\n";
        tmp &= f.saveMatrix(generationProbabilty.data(), maxGenerations);

        fs << "currentUniques[" << maxGenerations << "]=\n";
        tmp &= f.saveMatrix(currentUniques.data(), maxGenerations);

        fs << "logselection[" << maxGenerations << "]=\n";
        tmp &= f.saveMatrix(logselection.data(), maxGenerations);

        fs << "acceptance[" << maxGenerations << "]=\n";
        tmp &= f.saveMatrix(acceptance.data(), maxGenerations);

        fs << "covariance[" << nDim << "][" << nDim << "]=\n";
        tmp &= f.saveMatrix(covariance.data(), nDim, nDim);

        fs << "meantheta[" << maxGenerations << "][" << nDim << "]=\n";
        tmp &= f.saveMatrix(meantheta.data(), maxGenerations, nDim);

        f.closeFile();
        return tmp;
    }
    return false;
}

bool runinfo::save(std::string const &fileName)
{
    return save(&fileName[0]);
}

void runinfo::print()
{
    UMUQMSG("\n----------------------------\n");
    UMUQMSG("Problem Dimension= ", nDim);
    UMUQMSG("Generation= ", currentGeneration);

    // Create an instance of the IO object
    umuq::io f;

    {
        // Define the printing format
        umuq::ioFormat umuqFormat = {"\n", "", "[\n", "]\n"};

        UMUQMSG("\nEach generation coefficient of variation=\n");
        f.setWidth(-1);
        f.printMatrix(CoefVar, currentGeneration, 1, umuqFormat);

        UMUQMSG("\nEach generation probabilty=\n");
        f.setWidth(-1);
        f.printMatrix(generationProbabilty, currentGeneration, 1, umuqFormat);

        UMUQMSG("\nEach generation number of unique sample points=\n");
        f.setWidth(-1);
        f.printMatrix(currentUniques, currentGeneration, 1, umuqFormat);

        UMUQMSG("\nEach generation log selection for computing evidence=\n");
        f.setWidth(-1);
        f.printMatrix(logselection, currentGeneration, 1, umuqFormat);

        UMUQMSG("\nThe logarithm of evidence is=[", std::accumulate(logselection.data(), logselection.data() + currentGeneration, double{}), "]\n");

        UMUQMSG("\nEach generation acceptance rate=\n");
        f.setWidth(-1);
        f.printMatrix(acceptance, currentGeneration, 1, umuqFormat);
    }

    {
        // Define the printing format
        umuq::ioFormat umuqFormat = {" ", "", "[", "]\n"};

        UMUQMSG("\nEach generation mean of running information=\n");
        f.setWidth(-1);
        f.printMatrix(meantheta, currentGeneration, nDim, umuqFormat);

        UMUQMSG("\nCovariance of running information at generation[", currentGeneration, "]=\n");
        f.setWidth(-1);
        f.printMatrix(covariance, nDim, nDim, umuqFormat);
    }

    UMUQMSG("\n----------------------------\n");
}

bool runinfo::load(const char *fileName)
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
        tmp &= f.loadMatrix(CoefVar.data(), maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix(generationProbabilty.data(), maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix(currentUniques.data(), maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix(logselection.data(), maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix(acceptance.data(), maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix(covariance.data(), nDim, nDim);

        tmp &= f.readLine();
        tmp &= f.loadMatrix(meantheta.data(), maxGenerations, nDim);

        f.closeFile();
        return tmp;
    }
    return false;
}

bool runinfo::load(std::string const &fileName)
{
    return load(&fileName[0]);
}

inline bool runinfo::setUniqueNumber(int const nUniques)
{
    if (currentGeneration < maxGenerations)
    {
        currentUniques[currentGeneration] = nUniques;
        return true;
    }
    UMUQFAILRETURN("Generation number is greater than the defined maximum number!");
}

inline bool runinfo::setAcceptanceRate(double const acceptanceRate)
{
    if (currentGeneration < maxGenerations)
    {
        acceptance[currentGeneration] = acceptanceRate;
        return true;
    }
    UMUQFAILRETURN("Generation number is greater than the defined maximum number!");
}

inline void runinfo::broadcast()
{
    if (torc_num_nodes() == 1)
    {
        return;
    }

#if HAVE_MPI == 1
    for (int i = 0; i < torc_num_nodes(); i++)
    {
        torc_create_ex(i * torc_i_num_workers(), 1, (void (*)())broadcastTask, 1,
                       1, MPIDatatype<long long>, CALL_BY_REF,
                       reinterpret_cast<long long>(this));
    }
    torc_waitall();
#endif // MPI
}

void runinfo::printSampleStatistics()
{
    UMUQMSG("----------------------------");
    umuq::io f;
    // Define the printing format
    umuq::ioFormat meanFormat = {" ", "", "Mean=[", "]\nSample covariance matrix=\n"};
    umuq::ioFormat covarianceFormat = {" ", "\n", "[", "]"};
    f.printMatrix(meantheta.data() + currentGeneration * nDim, 1, nDim, meanFormat);
    f.printMatrix(covariance, nDim, nDim, covarianceFormat);
    UMUQMSG("----------------------------");
}

void broadcastTask(long long const other)
{
#if HAVE_MPI == 1
    auto obj = reinterpret_cast<runinfo *>(other);

    int const nDim(obj->nDim * obj->nDim);
    int const maxGenerations(obj->maxGenerations);

    MPI_Request request[3];

    MPI_Ibcast(obj->covariance.data(), nDim, MPIDatatype<double>, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Ibcast(obj->generationProbabilty.data(), maxGenerations, MPIDatatype<double>, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Ibcast(&obj->currentGeneration, 1, MPI_INT, 0, MPI_COMM_WORLD, &request[2]);

    MPI_Waitall(3, request, MPI_STATUSES_IGNORE);
#endif // MPI
}

} // namespace tmcmc
} // namespace umuq

#endif //UMUQ_RUNINFO
