#ifndef UMUQ_DATABASE_H
#define UMUQ_DATABASE_H

#include "core/core.hpp"
#include "mpidatatype.hpp"
#include "misc/arraywrapper.hpp"
#include "io/io.hpp"

namespace umuq
{

namespace tmcmc
{

/*!
 * \ingroup TMCMC_Module
 * 
 * \brief Updating the data information at each point SamplePoints 
 * 
 * \tparam T Data type (T is a floating-point type)
 * 
 * \param other          Database object which is casted to long long
 * \param SamplePoints   Points or sampling points array
 * \param FunValue       Function value at the sampling point 
 * \param DataArray      Array of data SamplePoints 
 * \param NDimDataArray  Dimension of G array
 * \param Surrogate      Surrogate
 */
template <typename T>
void updateDataTask(long long const other, T const *SamplePoints, T const *FunValue, T const *DataArray, int const *NDimDataArray, int const *Surrogate);

/*!
 * \ingroup TMCMC_Module
 * 
 * \brief A polymorphic function wrapper type for updateTask
 * 
 * \tparam T Data type 
 */
template <typename T>
using UPDATETASKTYPE = void (*)(long long const, T const *, T const *, T const *, int const *, int const *);

/*!
 * \ingroup TMCMC_Module
 * 
 * \brief It is True if updateDataTask has been registered, and false otherwise (logical).
 * 
 * \tparam T 
 */
template <typename T>
static bool isUpdateTaskRegistered = false;

/*!
 * \ingroup TMCMC_Module
 * 
 * \brief Mutex object
 * 
 */
static std::mutex updateTask_m;

} // namespace tmcmc

namespace tmcmc
{

/*! \class database
 * \ingroup TMCMC_Module
 *
 * \brief basic data base
 * 
 * \tparam T Data type (T is a floating-point type)
 *
 * \param samplePoints       Array for points in space
 * \param nDimSamplePoints   An integer argument shows the size of samplePoints
 * \param dataArray          Array of data
 * \param nDimDataArray      An integer argument shows the size of dataArray
 * \param fValue             Function value
 * \param surrogate          An integer argument shows the surrogate model
 * \param nSelection         An integer argument for selection of leaders only
 */
template <typename T>
class database
{
  public:
    /*!
     * \brief Construct a new database object
     * 
     */
    database();

    /*!
     * \brief Construct a new database object
     * 
     * \param nDim   Dimension of space (points)
     * \param nSize  Number of points 
     */
    database(int const nDim, int const nSize);

    /*!
     * \brief Construct a new database object
     * 
     * \param nDim1  Dimension of space (points)
     * \param nDim2  Dimension of the second array which could be prior
     * \param nSize  Number of points
     */
    database(int const nDim1, int const nDim2, int const nSize);

    /*!
     * \brief Move constructor, construct a new database object from input database object
     * 
     * \param other  Input database object
     */
    database(database<T> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other 
     * \return database<T>& 
     */
    database<T> &operator=(database<T> &&other);

    /*!
     * \brief Destroy the database object
     * 
     */
    ~database();

    /*!
     * \brief Reset the database class size
     * 
     * \param nSize  Number of sampling points (the new size of the database)
     * 
     * \return true 
     * \return false If there is not enough memory
     */
    bool reset(int const nSize);

    /*!
     * \brief Register task on the TORC task library
     * Before calling this function one should set the external functor otherwise it would crash!
     * 
     * \return true 
     * \return false if Task pointer is not correctly assigned
     */
    bool registerTask();

    /*!
     * \brief set the Task pointer to an external functor
     * 
     * \param func 
     */
    inline void setTask(UPDATETASKTYPE<T> const &func);

    /*!
     * \brief Swaps two database objects
     *
     * \param other Input database object
     */
    void swap(database<T> &other);

    /*!
     * \brief Get the size of database (in terms of number of entries)
     *
     * \return Size of the database (in terms of number of entries)
     */
    inline int size() const;

    /*!
     * \brief Reset the number of database entries to zero
     * 
     */
    inline void resetSize();

    /**
     * \brief Breaking the long chain length 
     * 
     * Here, we break the long chain length assigned to important samples to reduce or 
     * avoid error coming from the multiple chains mixing (default is 1 for BASIS TMCMC)
     * 
     * Reference:
     * Bayesian Annealed Sequential Importance Sampling: An Unbiased Version of Transitional Markov Chain Monte Carlo, 
     * ASCE-ASME JRU, 4(1), pp-011008, (2018)
     * 
     * \param minLength  Minimum chain length 
     * \param maxLength  Maximum chain length
     * 
     * \returns true 
     * \returns false If it encounters any problem
     */
    bool resetSelection(int const minLength = 1, int const maxLength = 1);

    /**
     * \brief Updating the data information on the new selections 
     * 
     * \param other datbase object 
     * 
     * \returns true 
     * \returns false If it encounters any problem
     */
    bool updateSelection(database<T> const &other);

    /**
     * \brief Update the work load on each queue based on the amount of work
     * It uses a greedy partitioning based on the amount of workload
     * 
     * \param nChains Number of chains
     * 
     * \returns true 
     * \returns false 
     */
    bool updateWorkload(int const nChains);

    /*!
     * \brief Printing the database data on the standard output
     *
     */
    bool print();

    /*!
     * \brief helper function for writing the data into a file
     *
     * Written data includes samplePoints, fValue, dataArray (if it exists)
     */
    bool save(const char *fname = "", int const IdNumber = 0);

    bool save(std::string const &fname, int const IdNumber = 0);

    /*!
     * \brief Helper function for loading the data from file
     *
     */
    bool load(const char *fname = "", int const IdNumber = 0);

    bool load(std::string const &fname, int const IdNumber = 0);

    /*!
     * \brief Updating the data information at each point SamplePoints 
     * 
     * \param SamplePoints   Points or sampling points array
     * \param FunValue       Function value at the sampling point
     * \param DataArray      Array of data SamplePoints 
     * \param NDimDataArray  Dimension of G array
     * \param Surrogate      Surrogate
     */
    void updateData(T const *SamplePoints, T const *FunValue, T const *DataArray, int const *NDimDataArray, int const *Surrogate);

    /*!
     * \brief Updating the data information at each point SamplePoints 
     * 
     * \param SamplePoints  Points or sampling points array
     * \param FunValue      Function value at the sampling point
     * \param DataArray     Array of data SamplePoints 
     * \param Surrogate     Surrogate
     */
    void update(T const *SamplePoints, T const FunValue, T const *DataArray = nullptr, int const Surrogate = std::numeric_limits<int>::max());

  private:
    /*!
     * \brief Set the List object
     *
     * \return true
     * \return false
     */
    inline bool resetList();

    /*!
     * \brief quick sort on list
     *
     */
    inline bool sort();

  private:
    // Make it noncopyable
    database(database<T> const &) = delete;

    // Make it not assignable
    database<T> &operator=(database<T> const &) = delete;

  private:
    /*! \class sortType
     *
     * \brief structure for sorting entires of database structure
     *
     * \param nSel An integer argument for selection of leaders only
     * \param idx  An intger argument for indexing
     */
    struct sortType
    {
        /*!
         * \brief Construct a new sort Type object
         * 
         */
        sortType();

        /*!
         * \brief Copy construct a new sort Type object
         * 
         * \param other 
         */
        sortType(sortType const &other);

        /*!
         * \brief Assignment constructor
         * 
         * \param other 
         * \return sortType& 
         */
        sortType &operator=(sortType const &other);

        //! Number of selections
        int nSel;
        //! Index number
        int idx;
    };

  public:
    //! Space dimension (Sampling points dimension)
    int nDimSamplePoints;

    //! Dimension of data array
    int nDimDataArray;

    //! Maximum number of data points
    int nSamplePoints;

    //! Index position
    std::size_t idxPosition;

    //! Points or sampling points array
    std::vector<T> samplePoints;

    //! Data array
    std::vector<T> dataArray;

    //! Function value
    std::vector<T> fValue;

    //! Surrogate
    std::vector<int> surrogate;

    //! Number of selection or Weighting number for each leader, for selection of leaders only
    std::vector<int> nSelection;

    //! Queue number for each point, for selection of leaders only
    std::vector<int> queue;

    //! Mutex object
    std::mutex m;

  private:
    //! Function pointer
    UPDATETASKTYPE<T> updateTask;

    //! List of sort data
    std::unique_ptr<sortType[]> list;
};

template <typename T>
database<T>::database() : nDimSamplePoints(0),
                          nDimDataArray(0),
                          nSamplePoints(0),
                          idxPosition(0),
                          updateTask(nullptr),
                          list(nullptr)
{
    if (!std::is_floating_point<T>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }

    setTask(updateDataTask);

    if (!registerTask())
    {
        UMUQFAIL("Failed to register the update task!");
    }
}

template <typename T>
database<T>::database(int const nDim, int const nSize) : nDimSamplePoints(nDim),
                                                         nDimDataArray(0),
                                                         nSamplePoints(0),
                                                         idxPosition(0),
                                                         updateTask(nullptr),
                                                         list(nullptr)
{
    if (!std::is_floating_point<T>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }

    if (!reset(nSize))
    {
        UMUQFAIL("Failed to initialize the data!");
    }

    setTask(updateDataTask);

    if (!registerTask())
    {
        UMUQFAIL("Failed to register the update task!");
    }
}

template <typename T>
database<T>::database(int const nDim1, int const nDim2, int const nSize) : nDimSamplePoints(nDim1),
                                                                           nDimDataArray(nDim2),
                                                                           nSamplePoints(0),
                                                                           idxPosition(0),
                                                                           updateTask(nullptr),
                                                                           list(nullptr)
{
    if (!std::is_floating_point<T>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }

    if (!reset(nSize))
    {
        UMUQFAIL("Failed to initialize the data!");
    }

    setTask(updateDataTask);

    if (!registerTask())
    {
        UMUQFAIL("Failed to register the update task!");
    }
}

template <typename T>
database<T>::database(database<T> &&other) : nDimSamplePoints(other.nDimSamplePoints),
                                             nDimDataArray(other.nDimDataArray),
                                             nSamplePoints(other.nSamplePoints),
                                             idxPosition(other.idxPosition),
                                             samplePoints(std::move(other.samplePoints)),
                                             dataArray(std::move(other.dataArray)),
                                             fValue(std::move(other.fValue)),
                                             surrogate(std::move(other.surrogate)),
                                             nSelection(std::move(other.nSelection)),
                                             queue(std::move(other.queue)),
                                             updateTask(std::move(other.updateTask)),
                                             list(std::move(other.list)) {}

template <typename T>
database<T> &database<T>::operator=(database<T> &&other)
{
    nDimSamplePoints = other.nDimSamplePoints;
    nDimDataArray = other.nDimDataArray;
    nSamplePoints = other.nSamplePoints;
    samplePoints = std::move(other.samplePoints);
    idxPosition = other.idxPosition;
    dataArray = std::move(other.dataArray);
    fValue = std::move(other.fValue);
    surrogate = std::move(other.surrogate);
    nSelection = std::move(other.nSelection);
    queue = std::move(other.queue);
    // m is default-initialized
    updateTask = std::move(other.updateTask);
    list = std::move(other.list);

    return *this;
}

template <typename T>
database<T>::~database() {}

template <typename T>
bool database<T>::reset(int const nSize)
{
    if (nSize < 0)
    {
        UMUQFAILRETURN("Wrong input size!");
    }

    nSamplePoints = nSize;

    if (nSamplePoints == 0)
    {
        UMUQWARNING("No entries -> Reseting the database object to size 0!");

        samplePoints.resize(0);
        dataArray.resize(0);
        fValue.resize(0);
        surrogate.resize(0);
        nSelection.resize(0);
        queue.resize(0);

        // Reset number of entries in the database
        idxPosition = 0;

        return true;
    }

    if (nDimSamplePoints > 0)
    {
        try
        {
            samplePoints.resize(nSamplePoints * nDimSamplePoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        samplePoints.resize(0);
    }

    if (nDimDataArray > 0)
    {
        try
        {
            dataArray.resize(nSamplePoints * nDimDataArray);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dataArray.resize(0);
    }

    try
    {
        fValue.resize(nSamplePoints);
        surrogate.resize(nSamplePoints, 0);
        nSelection.resize(nSamplePoints, 0);
        queue.resize(nSamplePoints, -1);
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }

    // Reset number of entries in the database
    idxPosition = 0;

    return true;
}

template <typename T>
bool database<T>::registerTask()
{
    {
        std::lock_guard<std::mutex> lock(updateTask_m);

        // Check if update task is already initialized
        if (isUpdateTaskRegistered<T>)
        {
            return true;
        }

        if (!updateTask)
        {
            UMUQFAILRETURN("Task Pointer is not assigned to the external function!");
        }
        isUpdateTaskRegistered<T> = true;
    }

    torc_register_task((void *)this->updateTask);

    return true;
}

template <typename T>
inline void database<T>::setTask(UPDATETASKTYPE<T> const &func)
{
    updateTask = func;
}

template <typename T>
void database<T>::swap(database<T> &other)
{
    std::unique_lock<std::mutex> lock_a(m, std::defer_lock);
    std::unique_lock<std::mutex> lock_b(other.m, std::defer_lock);
    std::lock(lock_a, lock_b);
    // Swap members
    std::swap(nDimSamplePoints, other.nDimSamplePoints);
    std::swap(nDimDataArray, other.nDimDataArray);
    std::swap(nSamplePoints, other.nSamplePoints);
    std::swap(idxPosition, other.idxPosition);
    samplePoints.swap(other.samplePoints);
    dataArray.swap(other.dataArray);
    fValue.swap(other.fValue);
    surrogate.swap(other.surrogate);
    nSelection.swap(other.nSelection);
    queue.swap(other.queue);
    std::swap(updateTask, other.updateTask);
    list.swap(other.list);
}

template <typename T>
inline int database<T>::size() const { return static_cast<int>(idxPosition); }

template <typename T>
inline void database<T>::resetSize() { idxPosition = 0; }

template <typename T>
bool database<T>::resetSelection(int const minLength, int const maxLength)
{
    if (minLength < 1 || maxLength < 1 || maxLength < minLength)
    {
        UMUQFAILRETURN("Wrong chain length size in the input!");
    }

    if (!resetList())
    {
        UMUQFAILRETURN("Failed to allocate list array!");
    }

    if (list)
    {
        // find out how many leaders we have
        int nLeaders(0);
        std::for_each(nSelection.begin(), nSelection.end(), [&](int const S_i) { nLeaders += static_cast<int>(S_i > 0); });

        {
            // Get the pointer
            int *nSel = nSelection.data();
            // set the counter to 0
            int idx(0);
            // Fill the list
            std::for_each(list.get(), list.get() + nSamplePoints, [&](sortType &l_i) {l_i.nSel = *nSel++; l_i.idx = idx++; });
        }

        if (this->sort())
        {
            int nChains(nLeaders);

            // Breaking long chains
            for (int i = 0; i < nLeaders; i++)
            {
                if (list[i].nSel > maxLength)
                {
                    while (list[i].nSel > maxLength)
                    {
                        list[nChains].nSel = maxLength;
                        list[nChains].idx = list[i].idx;
                        list[i].nSel -= maxLength;
                        nChains++;
                    }
                }
            }

            idxPosition = static_cast<std::size_t>(nChains);

            // If the maximum chain length is 1, we do not need to sort it
            if (maxLength != 1)
            {
                if (!this->sort())
                {
                    UMUQFAILRETURN("Failed to sort list array!");
                }
            }

            if (minLength != 1)
            {
                // Increasing the chain size
                for (int i = 0; i < nChains; i++)
                {
                    if (list[i].nSel && list[i].nSel < minLength)
                    {
                        list[i].nSel = minLength;
                    }
                }

                if (!this->sort())
                {
                    UMUQFAILRETURN("Failed to sort list array!");
                }
            }

            return true;
        }
        UMUQFAILRETURN("Failed to sort list array!");
    }
    UMUQFAILRETURN("List is not set yet!");
}

template <typename T>
bool database<T>::updateSelection(database<T> const &other)
{
    if (list)
    {
        if (nDimSamplePoints != other.nDimSamplePoints)
        {
            for (std::size_t i = 0; i < idxPosition; i++)
            {
                int const j = list[i].idx;

                T *From = other.samplePoints.data() + j * nDimSamplePoints;
                T *To = samplePoints.data() + i * nDimSamplePoints;

                std::copy(From, From + nDimSamplePoints, To);

                fValue[i] = other.fValue[j];

                nSelection[i] = list[i].nSel;
            }

            // Release the memory
            if (resetList())
            {
                return updateWorkload(static_cast<int>(idxPosition));
            }
            UMUQFAILRETURN("Failed to free list array!");
        }
        UMUQFAILRETURN("Input sampling points have a different dimension than the current points!");
    }
    UMUQFAILRETURN("List is not set yet!");
}

template <typename T>
bool database<T>::updateWorkload(int const nChains)
{
    if (nChains <= nSamplePoints)
    {
        // Greedy partitioning based on the workload
        int const nWorkers = torc_num_workers();

        // Dummy work load array
        std::vector<int> workLoad(nWorkers, 0);

        for (int i = 0; i < nChains; i++)
        {
            // Find the index of the work with minimum work load
            int const minLoadWorker = static_cast<int>(std::distance(workLoad.begin(), std::min_element(workLoad.begin(), workLoad.end())));

            // Chain i would be assigned to that queue
            queue[i] = minLoadWorker;

            // Sum up the new load on the current work load
            workLoad[minLoadWorker] += nSelection[i];
        }

        return true;
    }
    UMUQFAILRETURN("Wrong number of leading chains!");
}

template <typename T>
bool database<T>::print()
{
    if (idxPosition > 0 && nDimSamplePoints > 0)
    {
        std::cout << "---- database priniting ----" << std::endl;

        umuq::io f;

        // Define the printing format
        umuq::ioFormat poFormat = {",", "", "POINT(", ") "};
        umuq::ioFormat fvFormat = {"", "", "FValue=", " "};
        umuq::ioFormat suFormat = {" ", "\n", "Surrogate=", ""};

        int const nCurrentSamplePoints = static_cast<int>(idxPosition);

        // Getting the maximum width in the data for nice printing
        int pWidth = f.getWidth<T>(samplePoints, nCurrentSamplePoints, nDimSamplePoints, std::cout);
        int fWidth = f.getWidth<T>(fValue, nCurrentSamplePoints, 1, std::cout);
        int Width = std::max<int>(pWidth, fWidth);
        int sWidth = f.getWidth<int>(surrogate, nCurrentSamplePoints, 1, std::cout);

        // Array wrapper on the data
        umuq::arrayWrapper<T> ParrayWrapper(samplePoints, nCurrentSamplePoints * nDimSamplePoints, nDimSamplePoints);
        umuq::arrayWrapper<T> FvalueWrapper(fValue, nCurrentSamplePoints);
        umuq::arrayWrapper<int> SurrogateWrapper(surrogate, nCurrentSamplePoints);

        auto fIt = FvalueWrapper.begin();
        auto sIt = SurrogateWrapper.begin();

        if (nDimDataArray > 0)
        {
            umuq::ioFormat gvFormat = {",", "\n", "dataArray=[", "]"};

            int gWidth = f.getWidth<T>(dataArray, nCurrentSamplePoints, nDimDataArray, std::cout);

            umuq::arrayWrapper<T> GarrayWrapper(dataArray, nCurrentSamplePoints * nDimDataArray, nDimDataArray);

            auto gIt = GarrayWrapper.begin();

            for (auto pIt = ParrayWrapper.begin(); pIt != ParrayWrapper.end(); pIt++)
            {
                f.setWidth(Width);
                f.printMatrix<T>(pIt.get(), nDimSamplePoints, fIt.get(), 1, 1, poFormat, fvFormat);
                f.setWidth(sWidth);
                f.printMatrix<int>(sIt.get(), suFormat);
                f.setWidth(gWidth);
                f.printMatrix<T>(gIt.get(), nDimDataArray, 1, gvFormat);
                fIt++;
                sIt++;
                gIt++;
            }
        }
        else
        {
            for (auto pIt = ParrayWrapper.begin(); pIt != ParrayWrapper.end(); pIt++)
            {
                f.setWidth(Width);
                f.printMatrix<T>(pIt.get(), nDimSamplePoints, fIt.get(), 1, 1, poFormat, fvFormat);
                f.setWidth(sWidth);
                f.printMatrix<int>(sIt.get(), suFormat);
                fIt++;
                sIt++;
            }
        }

        std::cout << "----------------------------" << std::endl;

        return true;
    }
    UMUQFAILRETURN("There is no sampling point information to print!");
}

template <typename T>
bool database<T>::save(const char *fname, int const IdNumber)
{
    if (idxPosition > 0 && nDimSamplePoints > 0)
    {
        char fileName[256];
        if (strlen(fname) == 0)
        {
            sprintf(fileName, "db_%03d.txt", IdNumber);
        }
        else
        {
            sprintf(fileName, "%s_%03d.txt", fname, IdNumber);
        }

        umuq::io f;
        if (f.openFile(fileName, f.out | f.trunc))
        {
            int const nCurrentSamplePoints = static_cast<int>(idxPosition);

            if (nDimDataArray > 0)
            {
                // Getting the maximum width in the data for nice printing
                {
                    int pWidth = f.getWidth<T>(samplePoints, nCurrentSamplePoints, nDimSamplePoints, f.getFstream());
                    int fWidth = f.getWidth<T>(fValue, nCurrentSamplePoints, 1, f.getFstream());
                    int Width = std::max<int>(pWidth, fWidth);
                    int gWidth = f.getWidth<T>(dataArray, nCurrentSamplePoints, nDimDataArray, f.getFstream());
                    Width = std::max<int>(Width, gWidth);

                    f.setWidth(Width);
                }

                T *tmp[3] = {samplePoints.data(), fValue.data(), dataArray.data()};
                int nCols[3] = {nDimSamplePoints, 1, nDimDataArray};

                if (!f.saveMatrix<T>(tmp, 3, nCols, 1, nCurrentSamplePoints))
                {
                    return false;
                }
            }
            else
            {
                // Getting the maximum width in the data for nice printing
                {
                    int pWidth = f.getWidth<T>(samplePoints, nCurrentSamplePoints, nDimSamplePoints, f.getFstream());
                    int fWidth = f.getWidth<T>(fValue, nCurrentSamplePoints, 1, f.getFstream());

                    int Width = std::max<int>(pWidth, fWidth);

                    f.setWidth(Width);
                }

                T *tmp[2] = {samplePoints.data(), fValue.data()};
                int nCols[2] = {nDimSamplePoints, 1};

                if (!f.saveMatrix<T>(tmp, 2, nCols, 1, nCurrentSamplePoints))
                {
                    return false;
                }
            }

            f.closeFile();

            return true;
        }
        UMUQFAILRETURN("Failed to open the file!");
    }
    UMUQFAILRETURN("There is no sampling point information to write!");
}

template <typename T>
bool database<T>::save(std::string const &fname, int const IdNumber)
{
    return save(&fname[0], IdNumber);
}

template <typename T>
bool database<T>::load(const char *fname, int const IdNumber)
{
    if (nSamplePoints > 0 && nDimSamplePoints > 0)
    {
        char fileName[256];
        if (strlen(fname) == 0)
        {
            sprintf(fileName, "db_%03d.txt", IdNumber);
        }
        else
        {
            sprintf(fileName, "%s_%03d.txt", fname, IdNumber);
        }

        umuq::io f;
        if (f.openFile(fileName, f.in))
        {
            if (nDimDataArray > 0)
            {
                T *tmp[3] = {samplePoints.data(), fValue.data(), dataArray.data()};
                int nCols[3] = {nDimSamplePoints, 1, nDimDataArray};

                if (!f.loadMatrix<T>(tmp, 3, nCols, 1, nSamplePoints))
                {
                    return false;
                }
            }
            else
            {
                T *tmp[2] = {samplePoints.data(), fValue.data()};
                int nCols[2] = {nDimSamplePoints, 1};

                if (!f.loadMatrix<T>(tmp, 2, nCols, 1, nSamplePoints))
                {
                    return false;
                }
            }

            f.closeFile();

            idxPosition = static_cast<std::size_t>(nSamplePoints);

            return true;
        }
        UMUQFAILRETURN("Failed to open the file!");
    }
    UMUQFAILRETURN("First you should create an instance of the database with correct size!");
}

template <typename T>
bool database<T>::load(std::string const &fname, int const IdNumber)
{
    return load(&fname[0], IdNumber);
}

template <typename T>
void database<T>::updateData(T const *SamplePoints, T const *FunValue, T const *DataArray, int const *NDimDataArray, int const *Surrogate)
{
    std::size_t pos;

    {
        std::lock_guard<std::mutex> lock(m);
        pos = idxPosition;
        idxPosition++;
    }

    if (pos < nSamplePoints)
    {
        std::copy(SamplePoints, SamplePoints + nDimSamplePoints, samplePoints.data() + pos * nDimSamplePoints);

        fValue[pos] = *FunValue;

        // NDimDataArray is just an indicator if we have DataArray input data or not
        if (*NDimDataArray > 0)
        {
            std::copy(DataArray, DataArray + nDimDataArray, dataArray.data() + pos * nDimDataArray);
        }

        if (*Surrogate < std::numeric_limits<int>::max())
        {
            surrogate[pos] = *Surrogate;
        }
    }
}

template <typename T>
void database<T>::update(T const *SamplePoints, T const FunValue, T const *DataArray, int const Surrogate)
{
    if (torc_node_id() == 0)
    {
        updateData(SamplePoints, &FunValue, DataArray, &nDimDataArray, &Surrogate);
        return;
    }

    int const nDimGarray1 = DataArray != nullptr;
    int const nDimGarray2 = nDimGarray1 ? nDimDataArray : 0;
    int const nDimSurroga = Surrogate < std::numeric_limits<int>::max();

    torc_create_direct(0, (void (*)())updateTask, 6,
                       1, MPIDatatype<long long>, CALL_BY_REF,
                       nDimSamplePoints, MPIDatatype<T>, CALL_BY_VAL,
                       1, MPIDatatype<T>, CALL_BY_VAL,
                       nDimGarray2, MPIDatatype<T>, CALL_BY_VAL,
                       nDimGarray1, MPI_INT, CALL_BY_VAL,
                       nDimSurroga, MPI_INT, CALL_BY_VAL,
                       reinterpret_cast<long long>(this), SamplePoints,
                       &FunValue, DataArray, &nDimDataArray, &Surrogate);

    // Do not kill the worker
    torc_waitall3();
}

template <typename T>
void updateDataTask(long long const other, T const *SamplePoints, T const *FunValue, T const *DataArray, int const *NDimDataArray, int const *Surrogate)
{
    auto obj = reinterpret_cast<database<T> *>(other);

    std::size_t pos;

    {
        std::lock_guard<std::mutex> lock(obj->m);
        pos = obj->idxPosition;
        obj->idxPosition++;
    }

    if (pos < obj->nSamplePoints)
    {
        std::copy(SamplePoints, SamplePoints + obj->nDimSamplePoints, obj->samplePoints.data() + pos * obj->nDimSamplePoints);

        obj->fValue[pos] = *FunValue;

        // NDimDataArray is just an indicator if we have DataArray input data or not
        if (*NDimDataArray > 0)
        {
            std::copy(DataArray, DataArray + obj->nDimDataArray, obj->dataArray.data() + pos * obj->nDimDataArray);
        }

        if (*Surrogate < std::numeric_limits<int>::max())
        {
            obj->surrogate[pos] = *Surrogate;
        }
    }
}

template <typename T>
inline bool database<T>::resetList()
{
    if (list)
    {
        list.reset(nullptr);
        return true;
    }
    else
    {
        try
        {
            list.reset(new sortType[nSamplePoints]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        return true;
    }
}

template <typename T>
inline bool database<T>::sort()
{
    if (list)
    {
        // Sort the list using standard library quick sort
        std::qsort(list.get(), nSamplePoints, sizeof(sortType), [](const void *p1, const void *p2) {
            sortType const *s1 = static_cast<sortType const *>(p1);
            sortType const *s2 = static_cast<sortType const *>(p2);
            /* -: ascending order, +: descending order */
            return (s2->nSel - s1->nSel);
        });
        return true;
    }
    UMUQFAILRETURN("List is not set for sorting!");
}

template <typename T>
umuq::tmcmc::database<T>::sortType::sortType() : nSel(0) {}

template <typename T>
umuq::tmcmc::database<T>::sortType::sortType(umuq::tmcmc::database<T>::sortType const &other) : nSel(other.nSel), idx(other.idx) {}

template <typename T>
typename umuq::tmcmc::database<T>::sortType &umuq::tmcmc::database<T>::sortType::operator=(umuq::tmcmc::database<T>::sortType const &other)
{
    nSel = other.nSel;
    idx = other.idx;
}

} // namespace tmcmc
} // namespace umuq

#endif // UMUQ_DATABASE
