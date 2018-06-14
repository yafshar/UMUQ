#ifndef UMUQ_BASIC_H
#define UMUQ_BASIC_H

#include "../core/core.hpp"
#include "../misc/array.hpp"
#include "../io/io.hpp"

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

/*! \class database
 *
 * \brief basic data base
 * 
 * \tparam T         Data type 
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
  database()
  {
    if (getType() == MPI_INT)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " This type is not supported in this class!" << std::endl;
      throw(std::runtime_error("Wrong type!"));
    }
  }

  /*!
   * \brief Construct a new database object
   * 
   * \param nDim   Dimension of space (points)
   * \param nSize  Number of points 
   */
  database(int nDim, int nSize)
  {
    database<T>::ndimParray = nDim;

    if (getType() == MPI_INT)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " This type is not supported in this class!" << std::endl;
      throw(std::runtime_error("Wrong type!"));
    }

    if (!init(nSize))
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
  database(int nDim1, int nDim2, int nSize)
  {
    database<T>::ndimParray = nDim1;
    database<T>::ndimGarray = nDim2;

    if (getType() == MPI_INT)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " This type is not supported in this class!" << std::endl;
      throw(std::runtime_error("Wrong type!"));
    }

    if (!init(nSize))
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
    database<T>::ndimParray = inputDB.ndimParray;
    database<T>::ndimGarray = inputDB.ndimGarray;
    database<T>::idxPos = inputDB.idxPos;
    database<T>::entries = inputDB.entries;
    database<T>::Parray = std::move(inputDB.Parray);
    database<T>::Garray = std::move(inputDB.Garray);
    database<T>::Fvalue = std::move(inputDB.Fvalue);
    database<T>::Surrogate = std::move(inputDB.Surrogate);
    database<T>::nSelection = std::move(inputDB.nSelection);
    database<T>::idxNumber = std::move(inputDB.idxNumber);
    database<T>::m = std::move(inputDB.m);
    database<T>::list = std::move(inputDB.list);
  }

  /*!
   * \brief Move assignment operator
   * 
   * \param inputDB 
   * \return database<T>& 
   */
  database<T> &operator=(database<T> &&inputDB)
  {
    database<T>::ndimParray = inputDB.ndimParray;
    database<T>::ndimGarray = inputDB.ndimGarray;
    database<T>::idxPos = inputDB.idxPos;
    database<T>::entries = inputDB.entries;
    database<T>::Parray = std::move(inputDB.Parray);
    database<T>::Garray = std::move(inputDB.Garray);
    database<T>::Fvalue = std::move(inputDB.Fvalue);
    database<T>::Surrogate = std::move(inputDB.Surrogate);
    database<T>::nSelection = std::move(inputDB.nSelection);
    database<T>::idxNumber = std::move(inputDB.idxNumber);
    database<T>::m = std::move(inputDB.m);
    database<T>::list = std::move(inputDB.list);

    return *this;
  }

  /*!
   * \brief Initialize the database class
   * 
   * \param nSize   Number of points
   * \return true   
   * \return false  If there is not enough memory
   */
  bool init(int nSize)
  {
    auto initialized(0);

    MPI_Initialized(&initialized);
    if (!initialized)
    {
      return false;
    }

    torc_register_task((void *)database<T>::update_Task1);
    torc_register_task((void *)database<T>::update_Task2);
    torc_register_task((void *)database<T>::update_Task3);
    torc_register_task((void *)database<T>::update_Task4);

    return reset(nSize);
  }

  /*!
   * \brief Reset the database class size
   * 
   * \param nSize   Number of points
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

    database<T>::entries = static_cast<std::size_t>(nSize);

    if (database<T>::ndimParray > 0)
    {
      try
      {
        database<T>::Parray.reset(new T[database<T>::entries * database<T>::ndimParray]);
      }
      catch (std::bad_alloc &e)
      {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        return false;
      }
    }

    if (database<T>::ndimGarray > 0)
    {
      try
      {
        database<T>::Garray.reset(new T[database<T>::entries * database<T>::ndimGarray]);
      }
      catch (std::bad_alloc &e)
      {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        return false;
      }
    }

    try
    {
      database<T>::Fvalue.reset(new T[database<T>::entries]);
      database<T>::Surrogate.reset(new int[database<T>::entries]());
      database<T>::nSelection.reset(new int[database<T>::entries]());
      database<T>::idxNumber.reset(new std::size_t[database<T>::entries]);
    }
    catch (std::bad_alloc &e)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
      return false;
    }

    std::iota(database<T>::idxNumber.get(), database<T>::idxNumber.get() + database<T>::entries, std::size_t{});

    return true;
  }

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
    if (IdxPos > database<T>::entries || IdxPos < 0)
    {
      return false;
    }
    database<T>::idxPos = static_cast<std::size_t>(IdxPos);
    return true;
  }

  /*!
   * \brief Swaps databses
   *
   * \param inputDB Input database object
   */
  void swap(database<T> &inputDB)
  {
    std::swap(database<T>::ndimParray, inputDB.ndimParray);
    std::swap(database<T>::ndimGarray, inputDB.ndimGarray);
    std::swap(database<T>::entries, inputDB.entries);
    database<T>::Parray.swap(inputDB.Parray);
    database<T>::Garray.swap(inputDB.Garray);
    database<T>::Fvalue.swap(inputDB.Fvalue);
    database<T>::Surrogate.swap(inputDB.Surrogate);
    database<T>::nSelection.swap(inputDB.nSelection);
    database<T>::idxNumber.swap(inputDB.idxNumber);
    database<T>::list.swap(inputDB.list);
    std::swap(database<T>::m, inputDB.m);
  }

  /*!
   * \brief Get the size of database
   *
   * \return Size of the database
   */
  inline std::size_t size() const { return database<T>::entries; }

  /*!
   * \brief quick sort
   *
   */
  inline bool sort()
  {
    //Allocate memory
    try
    {
      list.reset(new sortType[database<T>::entries]);
    }
    catch (std::bad_alloc &e)
    {
      std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
      std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
      return false;
    }

    //Get the pointer
    int *nsel = database<T>::nSelection.get();
    std::size_t *idx = database<T>::idxNumber.get();

    //Fill the list
    std::for_each(list.get(), list.get() + database<T>::entries, [&](sortType &si) {si.nsel = *nsel++; si.idx = *idx++; });

    //Sort the list using standard library quick sort
    std::qsort(list.get(), database<T>::entries, sizeof(sortType), [](const void *p1, const void *p2) {
      sortType const *s1 = static_cast<sortType const *>(p1);
      sortType const *s2 = static_cast<sortType const *>(p2);
      /* -: ascending order, +: descending order */
      return (s2->nsel - s1->nsel);
    });

    //Get the pointer
    nsel = database<T>::nSelection.get();
    idx = database<T>::idxNumber.get();

    //Correct the arrays based on the sorted list
    std::for_each(list.get(), list.get() + database<T>::entries, [&](sortType &si) {*nsel = si.nsel; *idx = si.idx; nsel++; idx++; });

    //Free the memory
    list.reset();
  }

  /*!
   * \brief Printing the database data on the standard output
   *
   */
  bool print()
  {
    if (database<T>::entries > 0 && database<T>::ndimParray > 0)
    {
      std::cout << "---- database priniting ----" << std::endl;

      io f;

      //Define the printing format
      ioFormat poFormat = {",", "", "POINT(", ") "};
      ioFormat fvFormat = {"", "", "Fvalue=", " "};
      ioFormat suFormat = {" ", "\n", "Surrogate=", ""};

      //Getting the maximum width in the data for nice printing
      int pWidth = f.getWidth<T>(database<T>::Parray, database<T>::entries, database<T>::ndimParray, std::cout);
      int fWidth = f.getWidth<T>(database<T>::Fvalue, database<T>::entries, 1, std::cout);
      int Width = std::max<int>(pWidth, fWidth);
      int sWidth = f.getWidth<int>(database<T>::Surrogate, database<T>::entries, 1, std::cout);

      //Array wrapper on the data
      ArrayWrapper<T> ParrayWrapper(database<T>::Parray, database<T>::entries * database<T>::ndimParray, database<T>::ndimParray);
      ArrayWrapper<T> FvalueWrapper(database<T>::Fvalue, database<T>::entries);
      ArrayWrapper<int> SurrogateWrapper(database<T>::Surrogate, database<T>::entries);

      auto fIt = FvalueWrapper.begin();
      auto sIt = SurrogateWrapper.begin();

      if (database<T>::ndimGarray > 0)
      {
        ioFormat gvFormat = {",", "\n", "Garray=[", "]"};

        int gWidth = f.getWidth<T>(database<T>::Garray, database<T>::entries, database<T>::ndimGarray, std::cout);

        ArrayWrapper<T> GarrayWrapper(database<T>::Garray, database<T>::entries * database<T>::ndimGarray, database<T>::ndimGarray);

        auto gIt = GarrayWrapper.begin();

        for (auto pIt = ParrayWrapper.begin(); pIt != ParrayWrapper.end(); pIt++)
        {
          f.setWidth(Width);
          f.printMatrix<T>(pIt.get(), database<T>::ndimParray, fIt.get(), 1, 1, poFormat, fvFormat);
          f.setWidth(sWidth);
          f.printMatrix<int>(sIt.get(), suFormat);
          f.setWidth(gWidth);
          f.printMatrix<T>(gIt.get(), database<T>::ndimGarray, 1, gvFormat);
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
          f.printMatrix<T>(pIt.get(), database<T>::ndimParray, fIt.get(), 1, 1, poFormat, fvFormat);
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
   */
  bool dump(int const IdNumber, const char *fname = "")
  {
    if (database<T>::entries > 0 && database<T>::ndimParray > 0)
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
        if (database<T>::ndimGarray > 0)
        {
          //Getting the maximum width in the data for nice printing
          {
            int pWidth = f.getWidth<T>(database<T>::Parray, database<T>::entries, database<T>::ndimParray, f.getFstream());
            int fWidth = f.getWidth<T>(database<T>::Fvalue, database<T>::entries, 1, f.getFstream());
            int Width = std::max<int>(pWidth, fWidth);
            int gWidth = f.getWidth<T>(database<T>::Garray, database<T>::entries, database<T>::ndimGarray, f.getFstream());
            Width = std::max<int>(Width, gWidth);

            f.setWidth(Width);
          }

          T *tmp[3] = {database<T>::Parray.get(), database<T>::Fvalue.get(), database<T>::Garray.get()};
          int nCols[3] = {database<T>::ndimParray, 1, database<T>::ndimGarray};

          if (!f.saveMatrix<T>(tmp, 3, nCols, 1, database<T>::entries))
          {
            return false;
          }
        }
        else
        {
          //Getting the maximum width in the data for nice printing
          {
            int pWidth = f.getWidth<T>(database<T>::Parray, database<T>::entries, database<T>::ndimParray, f.getFstream());
            int fWidth = f.getWidth<T>(database<T>::Fvalue, database<T>::entries, 1, f.getFstream());

            int Width = std::max<int>(pWidth, fWidth);

            f.setWidth(Width);
          }

          T *tmp[2] = {database<T>::Parray.get(), database<T>::Fvalue.get()};
          int nCols[2] = {database<T>::ndimParray, 1};

          if (!f.saveMatrix<T>(tmp, 2, nCols, 1, database<T>::entries))
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

  /*!
   * /brief Helper function for loading the data from file
   *
   */
  bool load(int const IdNumber, const char *fname = "")
  {
    if (database<T>::entries > 0 && database<T>::ndimParray > 0)
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
        if (database<T>::ndimGarray > 0)
        {
          T *tmp[3] = {database<T>::Parray.get(), database<T>::Fvalue.get(), database<T>::Garray.get()};
          int nCols[3] = {database<T>::ndimParray, 1, database<T>::ndimGarray};

          if (!f.loadMatrix<T>(tmp, 3, nCols, 1, database<T>::entries))
          {
            return false;
          }
        }
        else
        {
          T *tmp[2] = {database<T>::Parray.get(), database<T>::Fvalue.get()};
          int nCols[2] = {database<T>::ndimParray, 1};

          if (!f.loadMatrix<T>(tmp, 2, nCols, 1, database<T>::entries))
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

  inline MPI_Datatype getType()
  {
    char name = typeid(T).name()[0];
    switch (name)
    {
    case 'f':
      return MPI_FLOAT;
      break;
    case 'd':
      return MPI_DOUBLE;
      break;
    case 'e':
      return MPI_DOUBLE;
      break;
    default:
      return MPI_INT;
      break;
    }
  }

  static void update_Task1(T const *iParray, T const *iFvalue, T const *iGarray, int const *iSurrogate)
  {
    std::size_t pos;

    pthread_mutex_lock(&database<T>::m);
    pos = database<T>::idxPos;
    database<T>::idxPos++;
    pthread_mutex_unlock(&database<T>::m);

    if (pos < database<T>::entries)
    {
      std::copy(iParray, iParray + database<T>::ndimParray, database<T>::Parray.get() + pos * database<T>::ndimParray);

      database<T>::Fvalue[pos] = *iFvalue;

      if (iGarray != nullptr)
      {
        std::copy(iGarray, iGarray + database<T>::ndimGarray, database<T>::Garray.get() + pos * database<T>::ndimGarray);
      }

      if (*iSurrogate < std::numeric_limits<int>::max())
      {
        database<T>::Surrogate[pos] = *iSurrogate;
      }
    }
  }

  static void update_Task2(T const *iParray, T const *iFvalue, int const *iSurrogate)
  {
    std::size_t pos;

    pthread_mutex_lock(&database<T>::m);
    pos = database<T>::idxPos;
    database<T>::idxPos++;
    pthread_mutex_unlock(&database<T>::m);

    if (pos < database<T>::entries)
    {
      std::copy(iParray, iParray + database<T>::ndimParray, database<T>::Parray.get() + pos * database<T>::ndimParray);

      database<T>::Fvalue[pos] = *iFvalue;

      database<T>::Surrogate[pos] = *iSurrogate;
    }
  }

  static void update_Task3(T const *iParray, T const *iFvalue, T const *iGarray)
  {
    std::size_t pos;

    pthread_mutex_lock(&database<T>::m);
    pos = database<T>::idxPos;
    database<T>::idxPos++;
    pthread_mutex_unlock(&database<T>::m);

    std::cout << pos << " " << iParray[0] << " " << iParray[1] << "  F " << *iFvalue << " G " << iGarray[0] << " " << iGarray[1] << std::endl;

    if (pos < database<T>::entries)
    {
      std::copy(iParray, iParray + database<T>::ndimParray, database<T>::Parray.get() + pos * database<T>::ndimParray);

      database<T>::Fvalue[pos] = *iFvalue;

      std::copy(iGarray, iGarray + database<T>::ndimGarray, database<T>::Garray.get() + pos * database<T>::ndimGarray);
    }
  }

  static void update_Task4(T const *iParray, T const *iFvalue)
  {
    std::size_t pos;

    pthread_mutex_lock(&database<T>::m);
    pos = database<T>::idxPos;
    database<T>::idxPos++;
    pthread_mutex_unlock(&database<T>::m);

    if (pos < database<T>::entries)
    {
      std::copy(iParray, iParray + database<T>::ndimParray, database<T>::Parray.get() + pos * database<T>::ndimParray);

      database<T>::Fvalue[pos] = *iFvalue;
    }
  }

  void update(T const *iParray, T const iFvalue, T const *iGarray = nullptr, int const iSurrogate = std::numeric_limits<int>::max())
  {
    // if (torc_node_id() == 0)
    // {
    //   update_Task1(iParray, &iFvalue, iGarray, &iSurrogate);
    //   return;
    // }

    if (iSurrogate < std::numeric_limits<int>::max())
    {
      if (iGarray != nullptr)
      {
        //Message to the database manager (separate process?)
        //or direct execution by server thread
        torc_create_direct(0, (void (*)())database<T>::update_Task1, 4,
                           database<T>::ndimParray, database<T>::getType(), CALL_BY_VAL,
                           1, database<T>::getType(), CALL_BY_VAL,
                           database<T>::ndimGarray, database<T>::getType(), CALL_BY_VAL,
                           1, MPI_INT, CALL_BY_VAL,
                           iParray, &iFvalue, iGarray, &iSurrogate);
      }
      else
      {
      }
    }
    else
    {
      if (iGarray != nullptr)
      {
      }
      else
      {
      }
    }

    // torc_waitall3();
  }

private:
  // Make it noncopyable
  database(database<T> const &) = delete;

  // Make it not assignable
  database<T> &operator=(database<T> const &) = delete;

public:
  //! Space dimension (Sampling points dimension)
  static int ndimParray;

  //! Dimension of G data array
  static int ndimGarray;

private:
  //! Index position
  static std::size_t idxPos;

public:
  //! Number of entries in the data
  static std::size_t entries;

  //! Points or sampling points array
  static std::unique_ptr<T[]> Parray;

  //! G data array
  static std::unique_ptr<T[]> Garray;

  //! Function value
  static std::unique_ptr<T[]> Fvalue;

  //! Surrogate
  static std::unique_ptr<int[]> Surrogate;

  //! Number of selection or Weighting number for each point
  static std::unique_ptr<int[]> nSelection;

  //! Index number for sorting
  static std::unique_ptr<std::size_t[]> idxNumber;

private:
  //! Mutex object
  static pthread_mutex_t m;

  //! List of sort data
  std::unique_ptr<sortType[]> list;
};

template <typename T>
int database<T>::ndimParray = 0;

template <typename T>
int database<T>::ndimGarray = 0;

template <typename T>
std::size_t database<T>::idxPos = 0;

template <typename T>
std::size_t database<T>::entries = 0;

template <typename T>
std::unique_ptr<T[]> database<T>::Parray;

template <typename T>
std::unique_ptr<T[]> database<T>::Garray;

template <typename T>
std::unique_ptr<T[]> database<T>::Fvalue;

template <typename T>
std::unique_ptr<int[]> database<T>::Surrogate;

template <typename T>
std::unique_ptr<int[]> database<T>::nSelection;

template <typename T>
std::unique_ptr<std::size_t[]> database<T>::idxNumber;

template <typename T>
pthread_mutex_t database<T>::m = PTHREAD_MUTEX_INITIALIZER;
#endif
