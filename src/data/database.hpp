#ifndef UMUQ_DATABASE_H
#define UMUQ_DATABASE_H

/*! \class database
* \brief data structure
*
* \tparam T type of data structure
*
* \param entries   An integer argument shows the size of entry
* \param entry     Wrapper to the actual data to provide std iterator functionality
* \param entryData Array of data container 
* \param m         A mutex object
*/
template <class T>
class database
{
  public:
    /*!
     * \brief constructor for the data structure
     *    
     */
    database()
    {
        pthread_mutex_init(&m, NULL);

        if (!init(0))
        {
            throw(std::runtime_error("Not Initialiazed!"));
        }
    }

    /*!
     * \brief constructor for the data structure - initialize initial entries to the provided value
     * 
     * \param nsize
     */
    explicit database(int nsize)
    {
        pthread_mutex_init(&m, NULL);

        if (!init(nsize))
        {
            throw(std::runtime_error("Not Initialiazed!"));
        }
    }

    /*!
     * \brief Move constructor
     * 
     * \param inputObj data to be moved
     */
    explicit database(database<T> &&inputObj)
    {
        entries = inputObj.entries;
        entry = std::move(inputObj.entry);
        entryData = std::move(inputObj.entryData);
        m = std::move(inputObj.m);
    }

    /*!
     * \brief Move assignment operator
     * 
     * \param inputObj
     */
    database<T> &operator=(database<T> &&inputObj)
    {
        entries = inputObj.entries;
        entry = std::move(inputObj.entry);
        entryData = std::move(inputObj.entryData);
        m = std::move(inputObj.m);

        return *this;
    }

    /*!
     * \brief access element at provided index 
     * 
     * \param id
     * 
     * \return element @(id)
     */
    inline T &operator()(std::size_t id)
    {
        return entry[id];
    }

    /*!
     * \brief access element at provided index with bound checking
     * 
     * \param id
     * 
     * \return element @(id)
     */
    inline T &at(std::size_t id)
    {
        if (id < entries)
        {
            return entry[id];
        }
        return T{};
    }

    /*!
     * \brief access element at provided index with bound checking
     * 
     * \param id
     * 
     * \return element @(id)
     */
    inline T &at(std::size_t id) const
    {
        if (id < entries)
        {
            return entry[id];
        }
        return T{};
    }

    /*!
     * \brief Initializes data with provided size with default value of used type
     * 
     * \param nsize1
     */
    bool init(int nsize1)
    {
        entries = nsize1;

        try
        {
            entryData.reset(new T[entries]);
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }

        T *array = entryData.get();

        entry.set(array, entries);

        return true;
    }

    /*!
     * \brief Swaps data of datas this <-> inputObj
     * 
     * \param inputObj
     */
    void swap(database<T> &inputObj)
    {
        std::swap(entries, inputObj.entries);
        entry.swap(inputObj.entry);

        entryData.swap(inputObj.entryData);
        std::swap(m, inputObj.m);
    }

  public:
    //! Number of entries in the data
    std::size_t entries;

    //! Wrapper to the actual data
    ArrayWrapper<T> entry;

  private:
    //! Array of data container
    std::unique_ptr<T[]> entryData;

    //! Mutex object
    pthread_mutex_t m;

  private:
    // make it noncopyable
    database(database<T> const &) = delete;

    // make it not assignable
    database<T> &operator=(database<T> const &) = delete;
};

#endif
