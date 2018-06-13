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
  database() : ndimParray(0),
               ndimGarray(0),
               idxPos(0),
               entries(0)
  {
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
                                  idxPos(0)
  {
    pthread_mutex_init(&m, NULL);

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
  database(int nDim1, int nDim2, int nSize) : ndimParray(nDim1),
                                              ndimGarray(nDim2),
                                              idxPos(0)
  {
    pthread_mutex_init(&m, NULL);

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
    list = std::move(inputDB.list);

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

    entries = static_cast<std::size_t>(nSize);

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
   * \brief Swaps databses
   * 
   * \param inputDB Input database object
   */
  void swap(database<T> &inputDB)
  {
    std::swap(ndimParray, inputDB.ndimParray);
    std::swap(ndimGarray, inputDB.ndimGarray);
    std::swap(entries, inputDB.entries);
    Parray.swap(inputDB.Parray);
    Garray.swap(inputDB.Garray);
    Fvalue.swap(inputDB.Fvalue);
    Surrogate.swap(inputDB.Surrogate);
    nSelection.swap(inputDB.nSelection);
    idxNumber.swap(inputDB.idxNumber);
    list.swap(inputDB.list);
    std::swap(m, inputDB.m);
  }

  /*!
   * \brief Get the size of database
   * 
   * \return Size of the database
   */
  inline std::size_t size() const { return entries; }

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
          f.printMatrix<T>(gIt.get(), gvFormat);
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
   */
  bool dump(int const IdNumber, const char *fname = "")
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

  /*!
   * /brief Helper function for loading the data from file
   *
   */
  bool load(int const IdNumber, const char *fname = "")
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

  /*!
   * \brief Update the data
   * 
   * \param  iParray     Input array of data (input sample points)
   * \param  iFvalue     Input function value
   * \param  iGarray     Input array of data
   * \param  iSurrogate  Input data for surrogate
   */
  void update_Task(T const *iParray, T const iFvalue, T const *iGarray = nullptr, int const iSurrogate = std::numeric_limits<int>::max())
  {
    std::size_t pos;

    pthread_mutex_lock(&m);
    pos = idxPos;
    idxPos++;
    pthread_mutex_unlock(&m);

    if (pos < entries)
    {
      std::copy(iParray, iParray + ndimParray, Parray.get() + pos * ndimParray);

      Fvalue[pos] = iFvalue;

      if (iGarray != nullptr)
      {
        std::copy(iGarray, iGarray + ndimGarray, Garray.get() + pos * ndimGarray);
      }

      if (iSurrogate < std::numeric_limits<int>::max())
      {
        Surrogate[pos] = iSurrogate;
      }
    }
  }

  void update(T const *iParray, T const iFvalue, T const *iGarray = nullptr, int const iSurrogate = std::numeric_limits<int>::max())
  {
    // if (torc_node_id() == 0)
    // {
    //   update_Task(iParray, iFvalue, iGarray, iSurrogate);
    //   return;
    // }

    // if (iGarray == nullptr)
    // {
    //   //Message to the database manager (separate process?) 
    //   //or direct execution by server thread
    //   torc_create_direct(0, (void (*)())update_Task, 3,
    //                      data.Nth, MPI_DOUBLE, CALL_BY_VAL,
    //                      1, MPI_DOUBLE, CALL_BY_COP,
    //                      1, MPI_INT, CALL_BY_COP,
    //                      point, &F, &surrogate);
    // }
    // else
    // {
    //   /* xxx: for CALL_BY_VAL: in the full-version of the library, with n=1 we had segv */
    //   torc_create_direct(0, (void (*)())torc_update_full_db_task_p5, 5,
    //                      data.Nth, MPI_DOUBLE, CALL_BY_VAL,
    //                      1, MPI_DOUBLE, CALL_BY_COP,
    //                      n, MPI_DOUBLE, CALL_BY_COP,
    //                      1, MPI_INT, CALL_BY_COP,
    //                      1, MPI_INT, CALL_BY_COP,
    //                      point, &F, G, &n, &surrogate);
    // }
    // torc_waitall3();
  }

private:
  // Make it noncopyable
  database(database<T> const &) = delete;

  // Make it not assignable
  database<T> &operator=(database<T> const &) = delete;

public:
  //! Space dimension (Sampling points dimension)
  int ndimParray;

  //! Dimension of G data array
  int ndimGarray;

private:
  //! Index position
  std::size_t idxPos;

public:
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

private:
  //! Mutex object
  pthread_mutex_t m;

  //! List of sort data
  std::unique_ptr<sortType[]> list;
};

#endif
