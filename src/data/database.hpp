#ifndef UMUQ_DATABASE_H
#define UMUQ_DATABASE_H

/*!
* \brief data structure
*
* \tparam T type of data structure
* \param entry
* \param entries an integer argument shows the size of entry
* \param m A mutex object
*/
template <class T>
class data
{
  public:
    ArrayWrapper<T> entry;
    size_t entries;

  private:
    std::unique_ptr<T[]> entryData;
    pthread_mutex_t m;

  public:
    /*!
     * \brief constructor for the data structure
     *    
     */
    data() { init(0); }

    /*!
     * \brief constructor for the data structure - initialize initial entries to the provided value
     * @param nsize
     */
    data(int nsize) { init(nsize); }
    
    /*!
     * \brief constructor for the data structure - initialize initial entries to the provided value
     * @param nsize
     * @param initData
     */
    
    data(int nsize, T initData) { init(nsize, initData); }
    /*!
     * Move constructor
     * @param inputObj data to be moved
     */
    
    data(data &&inputObj)
    {
        entries = inputObj.entries;
        entry = std::move(inputObj.entry);
        entryData = std::move(inputObj.entryData);
    }
    
    /*!
     * Move assignment operator
     * @param inputObj
     */
    data &operator=(data &&inputObj)
    {
        entries = inputObj.entries;
        entry = std::move(inputObj.entry);
        entryData = std::move(inputObj.entryData);
        return *this;
    }
    
    /*!
     * Constructor - initialize data with other data (data are copied and casted if needed).
     * @tparam U new type of data
     * @param initData input data
     */
    template <typename U>
    data(const data<U> &initData)
    {
        init(initData.entries);
        std::copy(initData.entry.begin(), initData.entry.end(), entry.begin());
    }
    
    /*!
     * Creates copy of this data converting each element to new type
     * @tparam U new type of data
     * @return created object by value
     */
    template <typename U>
    data<U> toType() const
    {
        data<U> new_value(entries);
        std::copy(entry.begin(), entry.end(), new_value.entry.begin());
        return new_value;
    }
    
    /*!
     * access element at provided index 
     * @param id
     * @return element @(id)
     */
    T &operator()(size_t id)
    {
        return entry[id];
    }

    /*!
     * access element at provided index
     * @param id
     * @return element @(id)
     */
    T &at(size_t id)
    {
        return entry[id];
    }

    /*!
     * access element at provided index
     * @param id
     * @return element @(id)
     */
    const T &at(size_t id) const
    {
        return entry[id];
    }

    /*!
     * Copies data from inputData requires prior initialization of 'this' object
     * @tparam U type of data
     * @param inputData input data with data
     */
    template <typename U>
    void copyFromentry(const data<U> &inputData)
    {
        std::copy(inputData.entry.begin(), inputData.entry.end(), entry.begin());
    }

    /*!
     * Initializes data with provided size with default value of used type
     * @param nsize1
     */
    bool init(int nsize1)
    {
        pthread_mutex_init(&m, NULL);

        entries = nsize1;
        entryData.reset(new T[nsize1]);

        if (entryData.get() == nullptr)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << nsize1 << std::endl;
            return false;
        }

        entry.set(entryData.get(), nsize1);

        return true;
    }

    /*!
     * Initilize data with provided size and initial value
     * @param nsize1
     * @param initData
     */
    bool init(int nsize1, T initData)
    {
        pthread_mutex_init(&m, NULL);

        entries = nsize1;
        entryData.reset(new T[nsize1]);

        T *array = entryData.get();
        if (array == nullptr)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << nsize1 << std::endl;
            return false;
        }

        entry.set(array, nsize1);

        std::fill(array, array + nsize1, initData);

        return true;
    }

    /*!
     * Initialize data with size taken from provided data
     * @tparam S
     * @param aInputMesh
     */
    template <typename U>
    bool init(const data<U> &initData)
    {
        return init(initData.entries);
    }

    /*!
     * Initializes data with size of 1/ratio of the provided size (rounding up if not divisible by ratio)
     * @param nsize1
     * @param ratio
     */
    bool initSize(int nsize1, double ratio)
    {
        const int nsize = std::ceil((double)nsize1 / ratio);
        return init(nsize);
    }

    /*!
     * Initializes data with size of 1/ratio of the provided size (rounding up if not divisible by ratio)
     * @param nsize1
     * @param ratio
     * @param initData
     */
    bool initSize(int nsize1, double ratio, T initData)
    {
        const int nsize = ceil((double)nsize1 / ratio);
        return init(nsize, initData);
    }

    /*!
     * Initializes data with size of 1/ratio of the provided size (rounding up if not divisible by ratio)
     * @param ratio
     * @param inputData
     */
    template <typename U>
    bool initSize(double ratio, const data<U> &inputData)
    {
        const int nsize = ceil((double)inputData.entries / ratio);
        return init(nsize);
    }

    /*!
     * Initializes data with size of 1/ratio of provided size (rounding up if not divisible by ratio) and initialize values
     * @tparam U
     * @param ratio
     * @param initData
     * @param inputData
     */
    template <typename U>
    bool initSize(double ratio, const data<U> &inputData, T initData)
    {
        const int nsize = ceil((double)inputData.entries / ratio);
        return init(nsize, initData);
    }

    /**
     * Swaps data of datas this <-> inputObj
     * @param inputObj
     */
    void swap(data &inputObj)
    {
        std::swap(entries, inputObj.entries);
        entryData.swap(inputObj.entryData);
        entry.swap(inputObj.entry);
    }

    friend std::ostream &operator<<(std::ostream &os, const data<T> &obj)
    {
        os << "data: size=" << obj.entries << " vSize:" << obj.entry.size() << " vCapacity:" << obj.entry.capacity() << " elementSize:" << sizeof(T);
        return os;
    }
};

#endif
