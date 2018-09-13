#ifndef UMUQ_DATABASE_H
#define UMUQ_DATABASE_H

#include "core/core.hpp"
#include "mpidatatype.hpp"
#include "misc/arraywrapper.hpp"
#include "io/io.hpp"

namespace umuq
{
/*! \namespace tmcmc
 * \brief Namespace containing all the functions for TMCMC algorithm
 *
 */
namespace tmcmc
{

/*!
 * \brief Updating the data information at each point @SamplePoints 
 * 
 * \tparam T Data type (T is a floating-point type)
 * 
 * \param other          Database object which is casted to long long
 * \param SamplePoints   Points or sampling points array
 * \param FunValue       Function value at the sampling point 
 * \param DataArray      Array of data @SamplePoints 
 * \param NDimDataArray  Dimension of G array
 * \param Surrogate      Surrogate
 */
template <typename T>
void update_Task(long long const other, T const *SamplePoints, T const *FunValue, T const *DataArray, int const *NDimDataArray, int const *Surrogate);

/*!
 * \brief A polymorphic function wrapper type for updateTask
 * 
 * \tparam T Data type 
 */
template <typename T>
using UPDATETASKTYPE = void (*)(long long const, T const *, T const *, T const *, int const *, int const *);

//! True if update_Task has been registered, and false otherwise (logical).
template <typename T>
static bool database_update_Task_registered = false;

//! Muex object
static std::mutex update_Task_m;

/*! \class database
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
 * \param nsel               An integer argument for selection of leaders only
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
    database(int nDim, int nSize);

    /*!
     * \brief Construct a new database object
     * 
     * \param nDim1  Dimension of space (points)
     * \param nDim2  Dimension of the second array which could be prior
     * \param nSize  Number of points
     */
    database(int nDim1, int nDim2, int nSize);

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
     * \param nSize   Number of points
     * 
     * \return true 
     * \return false  If there is not enough memory
     */
    bool reset(int nSize);

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
     * \brief Swaps databses
     *
     * \param other Input database object
     */
    void swap(database<T> &other);

    /*!
     * \brief Get the size of database in terms of its number of entries
     *
     * \return Size of the database in terms of its number of entries
     */
    inline std::size_t getSize() const;

    /*!
     * \brief Get the Index of the current position @idxPos
     *
     * \return the Index of the current position @idxPos
     */
    inline std::size_t getIndex() const;

    /*!
     * \brief reset the Index position to a @IdxPos position
     *
     * \param IdxPos
     *
     * \return true
     * \return false  If it is out of number of entries
     */
    inline bool resetIdxPos(int IdxPos);

    /*!
     * \brief quick sort
     *
     */
    inline bool sort();

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
     * \brief Updating the data information at each point @SamplePoints 
     * 
     * \param SamplePoints   Points or sampling points array
     * \param FunValue       Function value at the sampling point
     * \param DataArray      Array of data @SamplePoints 
     * \param NDimDataArray  Dimension of G array
     * \param Surrogate      Surrogate
     */
    void updateTask(T const *SamplePoints, T const *FunValue, T const *DataArray, int const *NDimDataArray, int const *Surrogate);

    /*!
     * \brief Updating the data information at each point @SamplePoints 
     * 
     * \param SamplePoints  Points or sampling points array
     * \param FunValue      Function value at the sampling point
     * \param DataArray     Array of data @SamplePoints 
     * \param Surrogate     Surrogate
     */
    void update(T const *SamplePoints, T const FunValue, T const *DataArray = nullptr, int const Surrogate = std::numeric_limits<int>::max());

  private:
    // Make it noncopyable
    database(database<T> const &) = delete;

    // Make it not assignable
    database<T> &operator=(database<T> const &) = delete;

    /*! \class sortType
     *
     * \brief structure for sorting entires of database structur
     *
     * \param nsel An integer argument for selection of leaders only
     * \param idx  An intger argument for indexing
     */
    struct sortType
    {
        int nsel;
        std::size_t idx;
    };

  public:
    //! Space dimension (Sampling points dimension)
    int nDimSamplePoints;

    //! Dimension of data array
    int nDimDataArray;

    //! Index position
    std::size_t idxPos;

    //! Number of entries in the data
    std::size_t entries;

    //! Points or sampling points array
    std::unique_ptr<T[]> samplePoints;

    //! Data array
    std::unique_ptr<T[]> dataArray;

    //! Function value
    std::unique_ptr<T[]> fValue;

    //! Surrogate
    std::unique_ptr<int[]> surrogate;

    //! Number of selection or Weighting number for each leader, for selection of leaders only
    std::unique_ptr<int[]> nSelection;

    //! Queue number for each point, for selection of leaders only
    std::unique_ptr<int[]> queue;

    //! Index number for sorting
    std::unique_ptr<std::size_t[]> idxNumber;

    //! Mutex object
    std::mutex m;

  private:
    //! Function pointer
    UPDATETASKTYPE<T> update_TaskP;

    //! List of sort data
    std::unique_ptr<sortType[]> list;
};

template <typename T>
database<T>::database() : nDimSamplePoints(0),
                          nDimDataArray(0),
                          idxPos(0),
                          entries(0),
                          update_TaskP(nullptr)
{
    if (!std::is_floating_point<T>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }

    setTask(update_Task);

    if (!registerTask())
    {
        UMUQFAIL("Failed to register the update task!");
    }
}

template <typename T>
database<T>::database(int nDim, int nSize) : nDimSamplePoints(nDim),
                                             nDimDataArray(0),
                                             idxPos(0),
                                             entries(0),
                                             update_TaskP(nullptr)
{
    if (!std::is_floating_point<T>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
    if (!reset(nSize))
    {
        UMUQFAIL("Failed to initialiaze the data!");
    }

    setTask(update_Task);

    if (!registerTask())
    {
        UMUQFAIL("Failed to register the update task!");
    }
}

template <typename T>
database<T>::database(int nDim1, int nDim2, int nSize) : nDimSamplePoints(nDim1),
                                                         nDimDataArray(nDim2),
                                                         idxPos(0),
                                                         entries(0),
                                                         update_TaskP(nullptr)
{
    if (!std::is_floating_point<T>::value)
    {
        UMUQFAIL("This type is not supported in this class!");
    }
    if (!reset(nSize))
    {
        UMUQFAIL("Failed to initialiaze the data!");
    }

    setTask(update_Task);

    if (!registerTask())
    {
        UMUQFAIL("Failed to register the update task!");
    }
}

template <typename T>
database<T>::database(database<T> &&other) : nDimSamplePoints(other.nDimSamplePoints),
                                             nDimDataArray(other.nDimDataArray),
                                             idxPos(other.idxPos),
                                             entries(other.entries),
                                             samplePoints(std::move(other.samplePoints)),
                                             dataArray(std::move(other.dataArray)),
                                             fValue(std::move(other.fValue)),
                                             surrogate(std::move(other.surrogate)),
                                             nSelection(std::move(other.nSelection)),
                                             queue(std::move(other.queue)),
                                             idxNumber(std::move(other.idxNumber)),
                                             update_TaskP(std::move(other.update_TaskP)),
                                             list(std::move(other.list))
{
    // m is default-initialized
}

template <typename T>
database<T> &database<T>::operator=(database<T> &&other)
{
    nDimSamplePoints = other.nDimSamplePoints;
    nDimDataArray = other.nDimDataArray;
    idxPos = other.idxPos;
    entries = other.entries;
    samplePoints = std::move(other.samplePoints);
    dataArray = std::move(other.dataArray);
    fValue = std::move(other.fValue);
    surrogate = std::move(other.surrogate);
    nSelection = std::move(other.nSelection);
    queue = std::move(other.queue);
    idxNumber = std::move(other.idxNumber);
    // m is default-initialized
    update_TaskP = std::move(other.update_TaskP);
    list = std::move(other.list);

    return *this;
}

template <typename T>
database<T>::~database() {}

template <typename T>
bool database<T>::reset(int nSize)
{
    if (nSize < 0)
    {
        UMUQFAILRETURN("Wrong input size!");
    }

    entries = static_cast<std::size_t>(nSize);

    if (entries == 0)
    {
        UMUQWARNING("No entries -> Reseting the databse object to size 0!");

        samplePoints.reset();
        dataArray.reset();
        fValue.reset();
        surrogate.reset();
        nSelection.reset();
        queue.reset();
        idxNumber.reset();

        return true;
    }

    if (nDimSamplePoints > 0)
    {
        try
        {
            samplePoints.reset(new T[entries * nDimSamplePoints]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        samplePoints.reset();
    }

    if (nDimDataArray > 0)
    {
        try
        {
            dataArray.reset(new T[entries * nDimDataArray]);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dataArray.reset();
    }

    try
    {
        fValue.reset(new T[entries]);
        surrogate.reset(new int[entries]());
        nSelection.reset(new int[entries]());
        queue.reset(new int[entries]());
        idxNumber.reset(new std::size_t[entries]);
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }

    std::iota(idxNumber.get(), idxNumber.get() + entries, std::size_t{});

    return true;
}

template <typename T>
bool database<T>::registerTask()
{
    {
        std::lock_guard<std::mutex> lock(update_Task_m);

        // Check if psrandom is already initilized
        if (database_update_Task_registered<T>)
        {
            return true;
        }

        if (!update_TaskP)
        {
            UMUQFAILRETURN("Task Pointer is not assigend to the external function!");
        }
        database_update_Task_registered<T> = true;
    }

    torc_register_task((void *)this->update_TaskP);

    return true;
}

template <typename T>
inline void database<T>::setTask(UPDATETASKTYPE<T> const &func)
{
    update_TaskP = func;
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
    std::swap(idxPos, other.idxPos);
    std::swap(entries, other.entries);
    samplePoints.swap(other.samplePoints);
    dataArray.swap(other.dataArray);
    fValue.swap(other.fValue);
    surrogate.swap(other.surrogate);
    nSelection.swap(other.nSelection);
    queue.swap(other.queue);
    idxNumber.swap(other.idxNumber);
    std::swap(update_TaskP, other.update_TaskP);
    list.swap(other.list);
}

template <typename T>
inline std::size_t database<T>::getSize() const { return entries; }

template <typename T>
inline std::size_t database<T>::getIndex() const { return idxPos; }

template <typename T>
inline bool database<T>::resetIdxPos(int IdxPos)
{
    if (IdxPos > entries || IdxPos < 0)
    {
        return false;
    }
    idxPos = static_cast<std::size_t>(IdxPos);
    return true;
}

template <typename T>
inline bool database<T>::sort()
{
    // Allocate memory
    try
    {
        list.reset(new sortType[entries]);
    }
    catch (...)
    {
        UMUQFAILRETURN("Failed to allocate memory!");
    }

    // Get the pointer
    int *nsel = nSelection.get();
    std::size_t *idx = idxNumber.get();

    // Fill the list
    std::for_each(list.get(), list.get() + entries, [&](sortType &si) {si.nsel = *nsel++; si.idx = *idx++; });

    // Sort the list using standard library quick sort
    std::qsort(list.get(), entries, sizeof(sortType), [](const void *p1, const void *p2) {
        sortType const *s1 = static_cast<sortType const *>(p1);
        sortType const *s2 = static_cast<sortType const *>(p2);
        /* -: ascending order, +: descending order */
        return (s2->nsel - s1->nsel);
    });

    // Get the pointer
    nsel = nSelection.get();
    idx = idxNumber.get();

    // Correct the arrays based on the sorted list
    std::for_each(list.get(), list.get() + entries, [&](sortType &si) {*nsel = si.nsel; *idx = si.idx; nsel++; idx++; });

    // Free the memory
    list.reset();
}

template <typename T>
bool database<T>::print()
{
    if (entries > 0 && nDimSamplePoints > 0)
    {
        std::cout << "---- database priniting ----" << std::endl;

        umuq::io f;

        // Define the printing format
        umuq::ioFormat poFormat = {",", "", "POINT(", ") "};
        umuq::ioFormat fvFormat = {"", "", "FValue=", " "};
        umuq::ioFormat suFormat = {" ", "\n", "Surrogate=", ""};

        // Getting the maximum width in the data for nice printing
        int pWidth = f.getWidth<T>(samplePoints, entries, nDimSamplePoints, std::cout);
        int fWidth = f.getWidth<T>(fValue, entries, 1, std::cout);
        int Width = std::max<int>(pWidth, fWidth);
        int sWidth = f.getWidth<int>(surrogate, entries, 1, std::cout);

        // Array wrapper on the data
        umuq::arrayWrapper<T> ParrayWrapper(samplePoints, entries * nDimSamplePoints, nDimSamplePoints);
        umuq::arrayWrapper<T> FvalueWrapper(fValue, entries);
        umuq::arrayWrapper<int> SurrogateWrapper(surrogate, entries);

        auto fIt = FvalueWrapper.begin();
        auto sIt = SurrogateWrapper.begin();

        if (nDimDataArray > 0)
        {
            umuq::ioFormat gvFormat = {",", "\n", "dataArray=[", "]"};

            int gWidth = f.getWidth<T>(dataArray, entries, nDimDataArray, std::cout);

            umuq::arrayWrapper<T> GarrayWrapper(dataArray, entries * nDimDataArray, nDimDataArray);

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
    return false;
}

template <typename T>
bool database<T>::save(const char *fname, int const IdNumber)
{
    if (entries > 0 && nDimSamplePoints > 0)
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
            if (nDimDataArray > 0)
            {
                // Getting the maximum width in the data for nice printing
                {
                    int pWidth = f.getWidth<T>(samplePoints, entries, nDimSamplePoints, f.getFstream());
                    int fWidth = f.getWidth<T>(fValue, entries, 1, f.getFstream());
                    int Width = std::max<int>(pWidth, fWidth);
                    int gWidth = f.getWidth<T>(dataArray, entries, nDimDataArray, f.getFstream());
                    Width = std::max<int>(Width, gWidth);

                    f.setWidth(Width);
                }

                T *tmp[3] = {samplePoints.get(), fValue.get(), dataArray.get()};
                int nCols[3] = {nDimSamplePoints, 1, nDimDataArray};

                if (!f.saveMatrix<T>(tmp, 3, nCols, 1, entries))
                {
                    return false;
                }
            }
            else
            {
                // Getting the maximum width in the data for nice printing
                {
                    int pWidth = f.getWidth<T>(samplePoints, entries, nDimSamplePoints, f.getFstream());
                    int fWidth = f.getWidth<T>(fValue, entries, 1, f.getFstream());

                    int Width = std::max<int>(pWidth, fWidth);

                    f.setWidth(Width);
                }

                T *tmp[2] = {samplePoints.get(), fValue.get()};
                int nCols[2] = {nDimSamplePoints, 1};

                if (!f.saveMatrix<T>(tmp, 2, nCols, 1, entries))
                {
                    return false;
                }
            }

            f.closeFile();

            return true;
        }
        return false;
    }
    return false;
}

template <typename T>
bool database<T>::save(std::string const &fname, int const IdNumber)
{
    return save(&fname[0], IdNumber);
}

template <typename T>
bool database<T>::load(const char *fname, int const IdNumber)
{
    if (entries > 0 && nDimSamplePoints > 0)
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
                T *tmp[3] = {samplePoints.get(), fValue.get(), dataArray.get()};
                int nCols[3] = {nDimSamplePoints, 1, nDimDataArray};

                if (!f.loadMatrix<T>(tmp, 3, nCols, 1, entries))
                {
                    return false;
                }
            }
            else
            {
                T *tmp[2] = {samplePoints.get(), fValue.get()};
                int nCols[2] = {nDimSamplePoints, 1};

                if (!f.loadMatrix<T>(tmp, 2, nCols, 1, entries))
                {
                    return false;
                }
            }

            f.closeFile();

            return true;
        }
        return false;
    }
    return false;
}

template <typename T>
bool database<T>::load(std::string const &fname, int const IdNumber)
{
    return load(&fname[0], IdNumber);
}

template <typename T>
void database<T>::updateTask(T const *SamplePoints, T const *FunValue, T const *DataArray, int const *NDimDataArray, int const *Surrogate)
{
    std::size_t pos;

    {
        std::lock_guard<std::mutex> lock(m);
        pos = idxPos;
        idxPos++;
    }

    if (pos < entries)
    {
        std::copy(SamplePoints, SamplePoints + nDimSamplePoints, samplePoints.get() + pos * nDimSamplePoints);

        fValue[pos] = *FunValue;

        //! NDimDataArray is just an indicator if we have DataArray input data or not
        if (*NDimDataArray > 0)
        {
            std::copy(DataArray, DataArray + nDimDataArray, dataArray.get() + pos * nDimDataArray);
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
        updateTask(SamplePoints, &FunValue, DataArray, &nDimDataArray, &Surrogate);
        return;
    }

    int const indimGarray1 = DataArray != nullptr;
    int const indimGarray2 = indimGarray1 ? nDimDataArray : 0;
    int const indimSurroga = Surrogate < std::numeric_limits<int>::max();

    torc_create_direct(0, (void (*)())update_TaskP, 6,
                       1, MPIDatatype<long long>, CALL_BY_REF,
                       nDimSamplePoints, MPIDatatype<T>, CALL_BY_VAL,
                       1, MPIDatatype<T>, CALL_BY_VAL,
                       indimGarray2, MPIDatatype<T>, CALL_BY_VAL,
                       indimGarray1, MPI_INT, CALL_BY_VAL,
                       indimSurroga, MPI_INT, CALL_BY_VAL,
                       reinterpret_cast<long long>(this), SamplePoints, &FunValue, DataArray, &nDimDataArray, &Surrogate);

    //! Do not kill the worker
    torc_waitall3();
}

template <typename T>
void update_Task(long long const other, T const *SamplePoints, T const *FunValue, T const *DataArray, int const *NDimDataArray, int const *Surrogate)
{
    auto obj = reinterpret_cast<database<T> *>(other);

    std::size_t pos;

    {
        std::lock_guard<std::mutex> lock(obj->m);
        pos = obj->idxPos;
        obj->idxPos++;
    }

    if (pos < obj->entries)
    {
        std::copy(SamplePoints, SamplePoints + obj->nDimSamplePoints, obj->samplePoints.get() + pos * obj->nDimSamplePoints);

        obj->fValue[pos] = *FunValue;

        // NDimDataArray is just an indicator if we have DataArray input data or not
        if (*NDimDataArray > 0)
        {
            std::copy(DataArray, DataArray + obj->nDimDataArray, obj->dataArray.get() + pos * obj->nDimDataArray);
        }

        if (*Surrogate < std::numeric_limits<int>::max())
        {
            obj->surrogate[pos] = *Surrogate;
        }
    }
}

} // namespace tmcmc
} // namespace umuq

#endif // UMUQ_DATABASE
