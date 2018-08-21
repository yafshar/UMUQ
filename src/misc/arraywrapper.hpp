#ifndef UMUQ_ARRAYWRAPPER_H
#define UMUQ_ARRAYWRAPPER_H

#include <iostream>
#include <iterator>

/*! \class arrayWrapper
  * \brief arrayWrapper is a class which creates a std iterator for an array of type T
  *
  * Expression of a class T (vector or matrix, or other types) as an array object
  */
template <class T>
class arrayWrapper
{
  public:
    /*! \class iterator
     *
     * \brief This class defines an iterator based on std::iterator
     * 
     */
    class iterator : public std::iterator<std::input_iterator_tag, T>
    {
      public:
        /*! 
         *  \brief iterator constructor
         *
         */
        iterator();

        /*!
         * \brief Construct a new iterator object
         * 
         * \param Stride 
         */
        explicit iterator(std::size_t Stride);

        /*!
         * \brief Construct a new iterator object
         * 
         * \param aPointer 
         */
        explicit iterator(T const *aPointer);

        /*!
         * \brief Construct a new iterator object
         * 
         * \param aPointer 
         * \param Stride 
         */
        iterator(T const *aPointer, std::size_t Stride);

        /*!
         * \brief Destroy the iterator object
         * 
         */
        ~iterator(){}; // nothing to do

        /*!
         * \brief Operator ==
         * 
         * \param rhs  
         * \return true 
         * \return false 
         */
        inline bool operator==(iterator const &rhs);

        /*!
         * \brief Operator !=
         * 
         * \param rhs 
         * \return true 
         * \return false 
         */
        inline bool operator!=(iterator const &rhs);

        /*!
         * \brief Operator ++
         * 
         * \return iterator& 
         */
        inline iterator &operator++();

        /*!
         * \brief Operator ++
         * 
         * \return iterator 
         */
        inline iterator operator++(int);

        /*!
         * \brief Access element at the current index 
         * 
         * \return Actual value at the current index 
         */
        inline T operator*();
        inline T const operator*() const;

        /*! 
         * \brief Get a pointer to the managed object or nullptr if no object is owned
         * 
         * \returns a pointer to the managed object or nullptr if no object is owned 
         */
        inline T *get() const;

      private:
        //! Iterator position
        T const *iPosition;

        //! Input stride
        std::size_t iStride;
    };

    /*! 
     *  \brief arrayWrapper constructor
     *
     */
    arrayWrapper();

    /*!
     * \brief Construct a new Array Wrapper object
     * 
     * \param InputArray    Input array data 
     * \param NumOfElements Size of the array 
     * \param Stride        Stride in the array elements
     */
    arrayWrapper(T const *InputArray, int const NumOfElements, int const Stride = 1);

    /*!
     * \brief Construct a new Array Wrapper object
     * 
     * \param InputArray    Input array data 
     * \param NumOfElements Size of the input array
     * \param Stride        Stride in the array elements
     */
    arrayWrapper(std::unique_ptr<T[]> const &InputArray, int const NumOfElements, int const Stride = 1);

    /*!
     * \brief Move constructor Construct a new Array Wrapper object
     * 
     * \param other 
     */
    arrayWrapper(arrayWrapper<T> &&other);

    /*!
     * \brief Move assignment 
     * 
     * \param other 
     * \return arrayWrapper& 
     */
    arrayWrapper<T> &operator=(arrayWrapper<T> &&other);

    /*!
     * \brief Destroy the Array Wrapper object
     * 
     */
    ~arrayWrapper(){};

    /*!
     * \brief set the wrapper
     * 
     * \param InputArray 
     * \param NumOfElements 
     */
    inline void set(T *InputArray, int const NumOfElements, int const Stride = 1);

    /*! 
     * \brief Returns an iterator to the beginning of Input
     * 
     * \return an iterator to the beginning of the given Input
     */
    inline iterator begin();
    inline iterator begin() const;

    /*! 
     * \brief Returns an iterator to the end
     * 
     * \return an iterator to the end of the given Input
     */
    inline iterator end();
    inline iterator end() const;

    /*!
     * \brief Get the size of array
     * 
     * \return Size of the array
     */
    inline std::size_t size() const;

    /*!
     * \brief Get the Stride
     * 
     * \return stride of the array
     */
    inline std::size_t stride() const;

    /*!
     * \brief Swap with the input arraywrapper object
     * 
     * \param InputArrayWrapperObj 
     */
    inline void swap(arrayWrapper<T> &InputArrayWrapperObj);

    /*!
     * \brief Access element at provided index @id with checking bounds
     * 
     * \param  id Requested index 
     * 
     * \returns Element @(id)
     */
    inline T at(int const id) const;

    /*!
     * \brief Access element at provided index
     * 
     * \param id Requested id
     * 
     * \returns Element @(id)
     */
    inline T operator()(int const id) const;

    /*!
     * \brief Access element at provided index
     * 
     * \param id Requested id
     * 
     * \returns Element @(id)
     */
    inline T operator[](int const id) const;

  private:
    // make it noncopyable
    arrayWrapper(arrayWrapper const &) = delete;

    // make it not assignable
    arrayWrapper &operator=(const arrayWrapper &) = delete;

  private:
    //! Pointer to the actual Input
    T const *iArray;

    //! Size of InputArray
    std::size_t iNumOfElements;

    //! stride
    std::size_t iStride;
};

/*! 
 *  \brief arrayWrapper constructor
 *
 */
template <class T>
arrayWrapper<T>::arrayWrapper() : iArray(nullptr), iNumOfElements(0), iStride(1) {}

/*!
 * \brief Construct a new Array Wrapper object
 * 
 * \param InputArray    Input array data 
 * \param NumOfElements Size of the array 
 * \param Stride        Stride in the array elements
 */
template <class T>
arrayWrapper<T>::arrayWrapper(T const *InputArray, int const NumOfElements, int const Stride) : iArray(InputArray),
                                                                                                iNumOfElements(NumOfElements),
                                                                                                iStride(Stride) {}
/*!
 * \brief Construct a new Array Wrapper object
 * 
 * \param InputArray    Input array data 
 * \param NumOfElements Size of the input array
 * \param Stride        Stride in the array elements
 */
template <class T>
arrayWrapper<T>::arrayWrapper(std::unique_ptr<T[]> const &InputArray, int const NumOfElements, int const Stride) : iArray(InputArray.get()),
                                                                                                                   iNumOfElements(NumOfElements),
                                                                                                                   iStride(Stride) {}

/*!
 * \brief Move constructor Construct a new Array Wrapper object
 * 
 * \param other 
 */
template <class T>
arrayWrapper<T>::arrayWrapper(arrayWrapper<T> &&other)
{
    this->iArray = std::move(other.iArray);
    this->iNumOfElements = other.iNumOfElements;
    this->iStride = other.iStride;
}

/*!
 * \brief Move assignment 
 * 
 * \param other 
 * \return arrayWrapper& 
 */
template <class T>
arrayWrapper<T> &arrayWrapper<T>::operator=(arrayWrapper<T> &&other)
{
    this->iArray = std::move(other.iArray);
    this->iNumOfElements = other.iNumOfElements;
    this->iStride = other.iStride;

    return *this;
}

/*!
 * \brief set the wrapper
 * 
 * \param InputArray 
 * \param NumOfElements 
 */
template <class T>
inline void arrayWrapper<T>::set(T *InputArray, int const NumOfElements, int const Stride)
{
    this->iArray = InputArray;
    this->iNumOfElements = NumOfElements >= 0 ? static_cast<std::size_t>(NumOfElements) : throw(std::runtime_error("Wrong index!"));
    this->iStride = Stride >= 0 ? static_cast<std::size_t>(Stride) : throw(std::runtime_error("Wrong index!"));
}

/*! 
 * \brief Returns an iterator to the beginning of Input
 * 
 * \return an iterator to the beginning of the given Input
 */
template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::begin()
{
    return this->iStride == 1 ? arrayWrapper<T>::iterator(this->iArray) : arrayWrapper<T>::iterator(this->iArray, this->iStride);
}

template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::begin() const
{
    return this->iStride == 1 ? arrayWrapper<T>::iterator(this->iArray) : arrayWrapper<T>::iterator(this->iArray, this->iStride);
}

/*! 
 * \brief Returns an iterator to the end
 * 
 * \return an iterator to the end of the given Input
 */
template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::end()
{
    return arrayWrapper<T>::iterator(this->iArray + this->iNumOfElements);
}

template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::end() const
{
    return arrayWrapper<T>::iterator(this->iArray + this->iNumOfElements);
}

/*!
 * \brief Get the size of array
 * 
 * \return Size of the array
 */
template <class T>
inline std::size_t arrayWrapper<T>::size() const { return this->iStride == 1 ? this->iNumOfElements : this->iNumOfElements / this->iStride; }

/*!
 * \brief Get the Stride
 * 
 * \return stride of the array
 */
template <class T>
inline std::size_t arrayWrapper<T>::stride() const { return this->iStride; }

/*!
 * \brief Swap with the input arraywrapper object
 * 
 * \param InputArrayWrapperObj 
 */
template <class T>
inline void arrayWrapper<T>::swap(arrayWrapper<T> &InputArrayWrapperObj)
{
    std::swap(this->iArray, InputArrayWrapperObj.iArray);
    std::swap(this->iNumOfElements, InputArrayWrapperObj.iNumOfElements);
    std::swap(this->iStride, InputArrayWrapperObj.iStride);
}

/*!
 * \brief Access element at provided index @id with checking bounds
 * 
 * \param  id Requested index 
 * 
 * \returns Element @(id)
 */
template <class T>
inline T arrayWrapper<T>::at(int const id) const
{
    std::size_t const i = id * this->iStride;
    return i < this->iNumOfElements ? this->iArray[i] : throw(std::runtime_error("Index out of bound!"));
}

/*!
 * \brief Access element at provided index
 * 
 * \param id Requested id
 * 
 * \returns Element @(id)
 */
template <class T>
inline T arrayWrapper<T>::operator()(int const id) const
{
    return this->iArray[id * this->iStride];
}

/*!
 * \brief Access element at provided index
 * 
 * \param id Requested id
 * 
 * \returns Element @(id)
 */
template <class T>
inline T arrayWrapper<T>::operator[](int const id) const
{
    return this->iArray[id * this->iStride];
}

/*! 
 *  \brief iterator constructor
 *
 */
template <class T>
arrayWrapper<T>::iterator::iterator() : iPosition(nullptr), iStride(1) {}

/*!
 * \brief Construct a new iterator object
 * 
 * \param Stride 
 */
template <class T>
arrayWrapper<T>::iterator::iterator(std::size_t Stride) : iPosition(nullptr), iStride(Stride) {}

/*!
 * \brief Construct a new iterator object
 * 
 * \param aPointer 
 */
template <class T>
arrayWrapper<T>::iterator::iterator(T const *aPointer) : iPosition(aPointer), iStride(1) {}

/*!
 * \brief Construct a new iterator object
 * 
 * \param aPointer 
 * \param Stride 
 */
template <class T>
arrayWrapper<T>::iterator::iterator(T const *aPointer, std::size_t Stride) : iPosition(aPointer), iStride(Stride) {}

/*!
 * \brief Operator ==
 * 
 * \param rhs  
 * \return true 
 * \return false 
 */
template <class T>
inline bool arrayWrapper<T>::iterator::operator==(arrayWrapper<T>::iterator const &rhs) { return this->iPosition == rhs.iPosition; }

/*!
 * \brief Operator !=
 * 
 * \param rhs 
 * \return true 
 * \return false 
 */
template <class T>
inline bool arrayWrapper<T>::iterator::operator!=(arrayWrapper<T>::iterator const &rhs) { return this->iStride == 1 ? this->iPosition != rhs.iPosition : this->iPosition < rhs.iPosition; }

/*!
 * \brief Operator ++
 * 
 * \return iterator& 
 */
template <class T>
inline typename arrayWrapper<T>::iterator &arrayWrapper<T>::iterator::operator++()
{
    this->iPosition += this->iStride;
    return *this;
}

/*!
 * \brief Operator ++
 * 
 * \return iterator 
 */
template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::iterator::operator++(int)
{
    arrayWrapper<T>::iterator t(*this);
    this->operator++();
    return t;
}

/*!
 * \brief Access element at the current index 
 * 
 * \return Actual value at the current index 
 */
template <class T>
inline T arrayWrapper<T>::iterator::operator*()
{
    return *this->iPosition;
}

template <class T>
inline T const arrayWrapper<T>::iterator::operator*() const
{
    return *this->iPosition;
}

/*! 
 * \brief Get a pointer to the managed object or nullptr if no object is owned
 * 
 * \returns a pointer to the managed object or nullptr if no object is owned 
 */
template <class T>
inline T *arrayWrapper<T>::iterator::get() const
{
    return const_cast<T *>(this->iPosition);
}

#endif // UMUQ_ARRAYWRAPPER
