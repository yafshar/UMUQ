#ifndef UMUQ_DATABASE_H
#define UMUQ_DATABASE_H

#include "../core/core.hpp"
#include "../misc/array.hpp"
#include "../io/io.hpp"
#include "mpidatatype.hpp"

/*!
 * \brief A polymorphic function wrapper type for update Task
 * 
 * \tparam T Data type 
 */
template <typename T>
using FUNCTIONPOINTER = void (*)(T const *, T const *, T const *, int const *, int const *);

/*! \class database
 *
 * \brief basic data base
 * 
 * \tparam T         Data type (T is a floating-point type)
 *
 * \param Parray     Array for points in space
 * \param ndimParray An integer argument shows the size of Parray
 * \param Garray     Array
 * \param ndimGarray An integer argument shows the size of Garray
 * \param Fvalue     Argument for the function value
 * \param surrogate  An integer argument shows the surrogate model
 * \param nsel       An integer argument for selection of leaders only
 */
template <typename T>
class database
{
public:
  /*!
   * \brief Construct a new database object
   * 
   */
  database() : ndimParray(0),
               ndimGarray(0),
               idxPos(0),
               entries(0),
               update_TaskP(nullptr)
  {
    if (!std::is_floating_point<T>::value)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " This type is not supported in this class!" << std::endl;
      throw(std::runtime_error("Wrong type!"));
    }

    pthread_mutex_init(&m, NULL);
  }

  /*!
   * \brief Construct a new database object
   * 
   * \param nDim   Dimension of space (points)
   * \param nSize  Number of points 
   */
  database(int nDim, int nSize) : ndimParray(nDim),
                                  ndimGarray(0),
                                  idxPos(0),
                                  entries(0),
                                  update_TaskP(nullptr)
  {
    if (!std::is_floating_point<T>::value)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " This class only supports float, double & long double type!" << std::endl;
      throw(std::runtime_error("Wrong type!"));
    }

    pthread_mutex_init(&m, NULL);

    if (!reset(nSize))
    {
      throw(std::runtime_error("Failed to initialiaze the data!"));
    }
  }

  /*!
   * \brief Construct a new database object
   * 
   * \param nDim1  Dimension of space (points)
   * \param nDim2  Dimension of the second array which could be prior
   * \param nSize  Number of points
   */
  database(int nDim1, int nDim2, int nSize) : ndimParray(nDim1),
                                              ndimGarray(nDim2),
                                              idxPos(0),
                                              entries(0),
                                              update_TaskP(nullptr)
  {
    if (!std::is_floating_point<T>::value)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " This type is not supported in this class!" << std::endl;
      throw(std::runtime_error("Wrong type!"));
    }

    pthread_mutex_init(&m, NULL);

    if (!reset(nSize))
    {
      throw(std::runtime_error("Failed to initialiaze the data!"));
    }
  }

  /*!
   * \brief Move constructor, construct a new database object from input database object
   * 
   * \param inputDB  Input database object
   */
  database(database<T> &&inputDB)
  {
    ndimParray = inputDB.ndimParray;
    ndimGarray = inputDB.ndimGarray;
    idxPos = inputDB.idxPos;
    entries = inputDB.entries;
    Parray = std::move(inputDB.Parray);
    Garray = std::move(inputDB.Garray);
    Fvalue = std::move(inputDB.Fvalue);
    Surrogate = std::move(inputDB.Surrogate);
    nSelection = std::move(inputDB.nSelection);
    idxNumber = std::move(inputDB.idxNumber);
    m = std::move(inputDB.m);
    update_TaskP = std::move(inputDB.update_TaskP);
    list = std::move(inputDB.list);
  }

  /*!
   * \brief Move assignment operator
   * 
   * \param inputDB 
   * \return database<T>& 
   */
  database<T> &operator=(database<T> &&inputDB)
  {
    ndimParray = inputDB.ndimParray;
    ndimGarray = inputDB.ndimGarray;
    idxPos = inputDB.idxPos;
    entries = inputDB.entries;
    Parray = std::move(inputDB.Parray);
    Garray = std::move(inputDB.Garray);
    Fvalue = std::move(inputDB.Fvalue);
    Surrogate = std::move(inputDB.Surrogate);
    nSelection = std::move(inputDB.nSelection);
    idxNumber = std::move(inputDB.idxNumber);
    m = std::move(inputDB.m);
    update_TaskP = std::move(inputDB.update_TaskP);
    list = std::move(inputDB.list);

    return *this;
  }

  /*!
   * \brief Destroy the database object
   * 
   */
  ~database() {}

  /*!
   * \brief Reset the database class size
   * 
   * \param nSize   Number of points
   * 
   * \return true 
   * \return false  If there is not enough memory
   */
  bool reset(int nSize)
  {
    if (nSize < 0)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " Wrong input size! " << std::endl;
      return false;
    }

    entries = static_cast<std::size_t>(nSize);

    if (entries == 0)
    {
      std::cerr << "Warning : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " No entries -> Reseting the databse object to size 0! " << std::endl;

      Parray.reset();
      Garray.reset();
      Fvalue.reset();
      Surrogate.reset();
      nSelection.reset();
      idxNumber.reset();

      return true;
    }

    if (ndimParray > 0)
    {
      try
      {
        Parray.reset(new T[entries * ndimParray]);
      }
      catch (std::bad_alloc &e)
      {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        return false;
      }
    }
    else
    {
      Parray.reset();
    }

    if (ndimGarray > 0)
    {
      try
      {
        Garray.reset(new T[entries * ndimGarray]);
      }
      catch (std::bad_alloc &e)
      {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        return false;
      }
    }
    else
    {
      Garray.reset();
    }

    try
    {
      Fvalue.reset(new T[entries]);
      Surrogate.reset(new int[entries]());
      nSelection.reset(new int[entries]());
      idxNumber.reset(new std::size_t[entries]);
    }
    catch (std::bad_alloc &e)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
      return false;
    }

    std::iota(idxNumber.get(), idxNumber.get() + entries, std::size_t{});

    return true;
  }

  /*!
   * \brief Initialize the database register task
   * Before calling this function one should set the external functor otherwise it would crash!
   *  
   * \returns false if Task pointer is not correctly assigned 
   */
  inline bool initTask()
  {
    return registerTask();
  }

  /*!
   * \brief Register task on the TORC task library
   * Before calling this function one should set the external functor otherwise it would crash!
   * 
   * \return true 
   * \return false if Task pointer is not correctly assigned
   */
  inline bool registerTask()
  {
    auto initialized(0);
    MPI_Initialized(&initialized);
    if (initialized)
    {
      if (update_TaskP != nullptr)
      {
        torc_register_task((void *)this->update_TaskP);

        return true;
      }
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " Task Pointer is not assigend to the external function! " << std::endl;
      return false;
    }
    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
    std::cerr << " MPI is not initialized! " << std::endl;
    std::cerr << " You should Initialize torc first! " << std::endl;
    return false;
  }

  /*!
   * \brief set the Task pointer to an external functor
   * 
   * \param func 
   */
  inline void setTask(FUNCTIONPOINTER<T> const &func)
  {
    update_TaskP = func;
  }

  /*!
   * \brief Swaps databses
   *
   * \param inputDB Input database object
   */
  void swap(database<T> &inputDB)
  {
    std::swap(ndimParray, inputDB.ndimParray);
    std::swap(ndimGarray, inputDB.ndimGarray);
    std::swap(idxPos, inputDB.idxPos);
    std::swap(entries, inputDB.entries);
    Parray.swap(inputDB.Parray);
    Garray.swap(inputDB.Garray);
    Fvalue.swap(inputDB.Fvalue);
    Surrogate.swap(inputDB.Surrogate);
    nSelection.swap(inputDB.nSelection);
    idxNumber.swap(inputDB.idxNumber);
    std::swap(m, inputDB.m);
    std::swap(update_TaskP, inputDB.update_TaskP);
    list.swap(inputDB.list);
  }

  /*!
   * \brief Get the size of database in terms of its number of entries
   *
   * \return Size of the database in terms of its number of entries
   */
  inline std::size_t getSize() const { return entries; }

  /*!
   * \brief Get the Index of the current position @idxPos
   *
   * \return the Index of the current position @idxPos
   */
  inline std::size_t getIndex() const { return idxPos; }

  /*!
   * \brief reset the Index position to a @IdxPos position
   *
   * \param IdxPos
   *
   * \return true
   * \return false  If it is out of number of entries
   */
  inline bool resetIdxPos(int IdxPos)
  {
    if (IdxPos > entries || IdxPos < 0)
    {
      return false;
    }
    idxPos = static_cast<std::size_t>(IdxPos);
    return true;
  }

  /*!
   * \brief quick sort
   *
   */
  inline bool sort()
  {
    //Allocate memory
    try
    {
      list.reset(new sortType[entries]);
    }
    catch (std::bad_alloc &e)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
      return false;
    }

    //Get the pointer
    int *nsel = nSelection.get();
    std::size_t *idx = idxNumber.get();

    //Fill the list
    std::for_each(list.get(), list.get() + entries, [&](sortType &si) {si.nsel = *nsel++; si.idx = *idx++; });

    //Sort the list using standard library quick sort
    std::qsort(list.get(), entries, sizeof(sortType), [](const void *p1, const void *p2) {
      sortType const *s1 = static_cast<sortType const *>(p1);
      sortType const *s2 = static_cast<sortType const *>(p2);
      /* -: ascending order, +: descending order */
      return (s2->nsel - s1->nsel);
    });

    //Get the pointer
    nsel = nSelection.get();
    idx = idxNumber.get();

    //Correct the arrays based on the sorted list
    std::for_each(list.get(), list.get() + entries, [&](sortType &si) {*nsel = si.nsel; *idx = si.idx; nsel++; idx++; });

    //Free the memory
    list.reset();
  }

  /*!
   * \brief Printing the database data on the standard output
   *
   */
  bool print()
  {
    if (entries > 0 && ndimParray > 0)
    {
      std::cout << "---- database priniting ----" << std::endl;

      io f;

      //Define the printing format
      ioFormat poFormat = {",", "", "POINT(", ") "};
      ioFormat fvFormat = {"", "", "Fvalue=", " "};
      ioFormat suFormat = {" ", "\n", "Surrogate=", ""};

      //Getting the maximum width in the data for nice printing
      int pWidth = f.getWidth<T>(Parray, entries, ndimParray, std::cout);
      int fWidth = f.getWidth<T>(Fvalue, entries, 1, std::cout);
      int Width = std::max<int>(pWidth, fWidth);
      int sWidth = f.getWidth<int>(Surrogate, entries, 1, std::cout);

      //Array wrapper on the data
      ArrayWrapper<T> ParrayWrapper(Parray, entries * ndimParray, ndimParray);
      ArrayWrapper<T> FvalueWrapper(Fvalue, entries);
      ArrayWrapper<int> SurrogateWrapper(Surrogate, entries);

      auto fIt = FvalueWrapper.begin();
      auto sIt = SurrogateWrapper.begin();

      if (ndimGarray > 0)
      {
        ioFormat gvFormat = {",", "\n", "Garray=[", "]"};

        int gWidth = f.getWidth<T>(Garray, entries, ndimGarray, std::cout);

        ArrayWrapper<T> GarrayWrapper(Garray, entries * ndimGarray, ndimGarray);

        auto gIt = GarrayWrapper.begin();

        for (auto pIt = ParrayWrapper.begin(); pIt != ParrayWrapper.end(); pIt++)
        {
          f.setWidth(Width);
          f.printMatrix<T>(pIt.get(), ndimParray, fIt.get(), 1, 1, poFormat, fvFormat);
          f.setWidth(sWidth);
          f.printMatrix<int>(sIt.get(), suFormat);
          f.setWidth(gWidth);
          f.printMatrix<T>(gIt.get(), ndimGarray, 1, gvFormat);
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
          f.printMatrix<T>(pIt.get(), ndimParray, fIt.get(), 1, 1, poFormat, fvFormat);
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

  /*!
   * /brief helper function for writing the data into a file
   *
   * Written data includes Parray, Fvalue, Garray (if it exists)
   */
  bool save(const char *fname = "", int const IdNumber = 0)
  {
    if (entries > 0 && ndimParray > 0)
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

      io f;
      if (f.openFile(fileName, f.out | f.trunc))
      {
        if (ndimGarray > 0)
        {
          //Getting the maximum width in the data for nice printing
          {
            int pWidth = f.getWidth<T>(Parray, entries, ndimParray, f.getFstream());
            int fWidth = f.getWidth<T>(Fvalue, entries, 1, f.getFstream());
            int Width = std::max<int>(pWidth, fWidth);
            int gWidth = f.getWidth<T>(Garray, entries, ndimGarray, f.getFstream());
            Width = std::max<int>(Width, gWidth);

            f.setWidth(Width);
          }

          T *tmp[3] = {Parray.get(), Fvalue.get(), Garray.get()};
          int nCols[3] = {ndimParray, 1, ndimGarray};

          if (!f.saveMatrix<T>(tmp, 3, nCols, 1, entries))
          {
            return false;
          }
        }
        else
        {
          //Getting the maximum width in the data for nice printing
          {
            int pWidth = f.getWidth<T>(Parray, entries, ndimParray, f.getFstream());
            int fWidth = f.getWidth<T>(Fvalue, entries, 1, f.getFstream());

            int Width = std::max<int>(pWidth, fWidth);

            f.setWidth(Width);
          }

          T *tmp[2] = {Parray.get(), Fvalue.get()};
          int nCols[2] = {ndimParray, 1};

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

  bool save(std::string const &fname = "", int const IdNumber = 0)
  {
    return save(&fname[0], IdNumber);
  }

  /*!
   * /brief Helper function for loading the data from file
   *
   */
  bool load(const char *fname = "", int const IdNumber = 0)
  {
    if (entries > 0 && ndimParray > 0)
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

      io f;
      if (f.openFile(fileName, f.in))
      {
        if (ndimGarray > 0)
        {
          T *tmp[3] = {Parray.get(), Fvalue.get(), Garray.get()};
          int nCols[3] = {ndimParray, 1, ndimGarray};

          if (!f.loadMatrix<T>(tmp, 3, nCols, 1, entries))
          {
            return false;
          }
        }
        else
        {
          T *tmp[2] = {Parray.get(), Fvalue.get()};
          int nCols[2] = {ndimParray, 1};

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

  bool load(std::string const &fname = "", int const IdNumber = 0)
  {
    return load(&fname[0], IdNumber);
  }


  /*!
   * \brief Updating the data information at each point @iParray 
   * 
   * \param iParray      Points or sampling points array
   * \param iFvalue      Function value at the sampling point
   * \param iGarray      Array of data @iParray 
   * \param indimGarray  Dimension of G array
   * \param iSurrogate   Surrogate
   */
  void updateTask(T const *iParray, T const *iFvalue, T const *iGarray, int const *indimGarray, int const *iSurrogate)
  {
    std::size_t pos;

    pthread_mutex_lock(&m);
    pos = idxPos;
    idxPos++;
    pthread_mutex_unlock(&m);

    if (pos < entries)
    {
      std::copy(iParray, iParray + ndimParray, Parray.get() + pos * ndimParray);

      Fvalue[pos] = *iFvalue;

      if (*indimGarray > 0)
      {
        std::copy(iGarray, iGarray + ndimGarray, Garray.get() + pos * ndimGarray);
      }

      if (*iSurrogate < std::numeric_limits<int>::max())
      {
        Surrogate[pos] = *iSurrogate;
      }
    }
  }

  /*!
   * \brief Updating the data information at each point @iParray 
   * 
   * \param iParray     Points or sampling points array
   * \param iFvalue     Function value at the sampling point
   * \param iGarray     Array of data @iParray 
   * \param iSurrogate  Surrogate
   */
  void update(T const *iParray, T const iFvalue, T const *iGarray = nullptr, int const iSurrogate = std::numeric_limits<int>::max())
  {
    if (torc_node_id() == 0)
    {
      updateTask(iParray, &iFvalue, iGarray, &ndimGarray, &iSurrogate);
      return;
    }

    int const indimGarray1 = iGarray != nullptr;
    int const indimGarray2 = indimGarray1 ? ndimGarray : 0;
    int const indimSurroga = iSurrogate < std::numeric_limits<int>::max();

    torc_create_direct(0, (void (*)())update_TaskP, 5,
                       ndimParray, MPIDatatype<T>, CALL_BY_VAL,
                       1, MPIDatatype<T>, CALL_BY_VAL,
                       indimGarray2, MPIDatatype<T>, CALL_BY_VAL,
                       indimGarray1, MPI_INT, CALL_BY_VAL,
                       indimSurroga, MPI_INT, CALL_BY_VAL,
                       iParray, &iFvalue, iGarray, &ndimGarray, &iSurrogate);

    torc_waitall3();
  }

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
  int ndimParray;

  //! Dimension of G data array
  int ndimGarray;

  //! Index position
  std::size_t idxPos;

  //! Number of entries in the data
  std::size_t entries;

  //! Points or sampling points array
  std::unique_ptr<T[]> Parray;

  //! G data array
  std::unique_ptr<T[]> Garray;

  //! Function value
  std::unique_ptr<T[]> Fvalue;

  //! Surrogate
  std::unique_ptr<int[]> Surrogate;

  //! Number of selection or Weighting number for each point
  std::unique_ptr<int[]> nSelection;

  //! Index number for sorting
  std::unique_ptr<std::size_t[]> idxNumber;

  //! Mutex object
  pthread_mutex_t m;

private:
  //! Function pointer
  FUNCTIONPOINTER<T> update_TaskP;

  //! List of sort data
  std::unique_ptr<sortType[]> list;
};

#endif
