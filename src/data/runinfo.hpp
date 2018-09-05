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

#endif //UMUQ_RUNINFO_H
