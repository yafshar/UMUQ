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
     * \param inputRI Input runinfo object
     * 
     */
    runinfo(runinfo<T> &&inputRI);

    /*!
     * \brief Move assignment operator
     * 
     * \param inputRI      Input runinfo object
     * \return runinfo<T>& 
     */
    runinfo<T> &operator=(runinfo<T> &&inputRI);

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
     * \param inputRI Input runinfo object
     */
    void swap(runinfo<T> &inputRI);

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
    if (!runinfo<T>::init())
    {
        UMUQFAIL("Failed to initialiaze!")
    }
}

/*!
 * \brief Move constructor, construct a new runinfo object from an input runinfo object
 * 
 * \param inputRI Input runinfo object
 * 
 */
template <typename T>
runinfo<T>::runinfo(runinfo<T> &&inputRI)
{
    runinfo<T>::nDim = inputRI.nDim;
    runinfo<T>::maxGenerations = inputRI.maxGenerations;
    runinfo<T>::Generation = inputRI.Generation;
    runinfo<T>::CoefVar = std::move(inputRI.CoefVar);
    runinfo<T>::p = std::move(inputRI.p);
    runinfo<T>::currentuniques = std::move(inputRI.currentuniques);
    runinfo<T>::logselection = std::move(inputRI.logselection);
    runinfo<T>::acceptance = std::move(inputRI.acceptance);
    runinfo<T>::SS = std::move(inputRI.SS);
    runinfo<T>::meantheta = std::move(inputRI.meantheta);
}

/*!
 * \brief Move assignment operator
 * 
 * \param inputRI      Input runinfo object
 * \return runinfo<T>& 
 */
template <typename T>
runinfo<T> &runinfo<T>::operator=(runinfo<T> &&inputRI)
{
    runinfo<T>::nDim = inputRI.nDim;
    runinfo<T>::maxGenerations = inputRI.maxGenerations;
    runinfo<T>::Generation = inputRI.Generation;
    runinfo<T>::CoefVar = std::move(inputRI.CoefVar);
    runinfo<T>::p = std::move(inputRI.p);
    runinfo<T>::currentuniques = std::move(inputRI.currentuniques);
    runinfo<T>::logselection = std::move(inputRI.logselection);
    runinfo<T>::acceptance = std::move(inputRI.acceptance);
    runinfo<T>::SS = std::move(inputRI.SS);
    runinfo<T>::meantheta = std::move(inputRI.meantheta);

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
        runinfo<T>::CoefVar.resize(runinfo<T>::maxGenerations, T{});
        runinfo<T>::p.resize(runinfo<T>::maxGenerations, T{});
        runinfo<T>::currentuniques.resize(runinfo<T>::maxGenerations, 0);
        runinfo<T>::logselection.resize(runinfo<T>::maxGenerations, T{});
        runinfo<T>::acceptance.resize(runinfo<T>::maxGenerations, T{});

        runinfo<T>::SS.resize(runinfo<T>::nDim * runinfo<T>::nDim);
        runinfo<T>::meantheta.resize(runinfo<T>::maxGenerations * runinfo<T>::nDim);
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }

    // set the first value to a high number
    runinfo<T>::CoefVar[0] = std::numeric_limits<T>::max();

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
    runinfo<T>::nDim = ProbDim;
    runinfo<T>::maxGenerations = MaxGenerations;
    return runinfo<T>::init();
}

/*!
 * \brief Exchanges the given runinfo object
 * 
 * \param inputRI Input runinfo object
 * 
 */
template <typename T>
void runinfo<T>::swap(runinfo<T> &inputRI)
{
    std::swap(runinfo<T>::nDim, inputRI.nDim);
    std::swap(runinfo<T>::maxGenerations, inputRI.maxGenerations);
    std::swap(runinfo<T>::Generation, inputRI.Generation);
    runinfo<T>::CoefVar.swap(inputRI.CoefVar);
    runinfo<T>::p.swap(inputRI.p);
    runinfo<T>::currentuniques.swap(inputRI.currentuniques);
    runinfo<T>::logselection.swap(inputRI.logselection);
    runinfo<T>::acceptance.swap(inputRI.acceptance);
    runinfo<T>::SS.swap(inputRI.SS);
    runinfo<T>::meantheta.swap(inputRI.meantheta);
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
    io f;
    if (f.openFile(fileName, f.out | f.trunc))
    {
        // Get the IO stream
        std::fstream &fs = f.getFstream();
        bool tmp = true;

        fs << std::fixed;

        fs << "ProblemDimension= " << runinfo<T>::nDim << "\n";
        fs << "maxGenerations= " << runinfo<T>::maxGenerations << "\n";
        fs << "Generation= " << runinfo<T>::Generation << "\n";

        fs << "CoefVar[" << runinfo<T>::maxGenerations << "]=\n";
        tmp &= f.saveMatrix<T>(runinfo<T>::CoefVar.data(), runinfo<T>::maxGenerations);

        fs << "p[" << runinfo<T>::maxGenerations << "]=\n";
        tmp &= f.saveMatrix<T>(runinfo<T>::p.data(), runinfo<T>::maxGenerations);

        fs << "currentuniques[" << runinfo<T>::maxGenerations << "]=\n";
        tmp &= f.saveMatrix<int>(runinfo<T>::currentuniques.data(), runinfo<T>::maxGenerations);

        fs << "logselection[" << runinfo<T>::maxGenerations << "]=\n";
        tmp &= f.saveMatrix<T>(runinfo<T>::logselection.data(), runinfo<T>::maxGenerations);

        fs << "acceptance[" << runinfo<T>::maxGenerations << "]=\n";
        tmp &= f.saveMatrix<T>(runinfo<T>::acceptance.data(), runinfo<T>::maxGenerations);

        fs << "SS[" << runinfo<T>::nDim << "][" << runinfo<T>::nDim << "]=\n";
        tmp &= f.saveMatrix<T>(runinfo<T>::SS.data(), runinfo<T>::nDim, runinfo<T>::nDim);

        fs << "meantheta[" << runinfo<T>::maxGenerations << "][" << nDim << "]=\n";
        tmp &= f.saveMatrix<T>(runinfo<T>::meantheta.data(), runinfo<T>::maxGenerations, runinfo<T>::nDim);

        f.closeFile();
        return tmp;
    }
    return false;
}

template <typename T>
bool runinfo<T>::save(std::string const &fileName)
{
    return runinfo<T>::save(&fileName[0]);
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

        if (ProbDim != runinfo<T>::nDim || MaxGenerations != runinfo<T>::maxGenerations)
        {
            if (!runinfo<T>::reset(ProbDim, MaxGenerations))
            {
                return false;
            }
        }

        tmp &= f.readLine();
        prs.parse(f.getLine().substr(12));
        runinfo<T>::Generation = prs.at<int>(0);

        tmp &= f.readLine();
        tmp &= f.loadMatrix<T>(runinfo<T>::CoefVar.data(), runinfo<T>::maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix<T>(runinfo<T>::p.data(), runinfo<T>::maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix<int>(runinfo<T>::currentuniques.data(), runinfo<T>::maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix<T>(runinfo<T>::logselection.data(), runinfo<T>::maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix<T>(runinfo<T>::acceptance.data(), runinfo<T>::maxGenerations);

        tmp &= f.readLine();
        tmp &= f.loadMatrix<T>(runinfo<T>::SS.data(), runinfo<T>::nDim, runinfo<T>::nDim);

        tmp &= f.readLine();
        tmp &= f.loadMatrix<T>(runinfo<T>::meantheta.data(), runinfo<T>::maxGenerations, runinfo<T>::nDim);

        f.closeFile();
        return tmp;
    }
    return false;
}

template <typename T>
bool runinfo<T>::load(std::string const &fileName)
{
    return runinfo<T>::load(&fileName[0]);
}

#endif //UMUQ_RUNINFO_H
