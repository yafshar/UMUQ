#ifndef UMUQ_ARRAYWRAPPER_H
#define UMUQ_ARRAYWRAPPER_H

#include <iostream>
#include <type_traits>

namespace umuq
{

/*! \class arrayWrapper
 *
 * \brief This is a class which creates std iterator like behavior for an array of class T.
 * 
 * Expression of a class T (vector or matrix, or other types) as an array object.
 * 
 * \tparam T Basic data type or vector, matrix, or other types
 */
template <class T>
class arrayWrapper
{
  public:
    /*! \class iterator
     *
     * \brief This class defines an iterator like on std::iterator.
     * 
     * The base class provided to simplify definitions of the required types for iterators. 
     * 
     */
    class iterator
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
         * \param Stride Stride in the input array of data
         */
        explicit iterator(std::size_t const Stride);

        /*!
         * \brief Construct a new iterator object
         * 
         * \param aPointer Pointer to the input array data 
         */
        explicit iterator(T *aPointer);

        /*!
         * \brief Construct a new iterator object
         * 
         * \param aPointer Pointer to the input array data 
         */
        explicit iterator(T const *aPointer);

        /*!
         * \brief Construct a new iterator object
         * 
         * \param aPointer  Pointer to the input array data 
         * \param Stride    Stride in the array of data
         */
        iterator(T *aPointer, std::size_t const Stride);

        /*!
         * \brief Construct a new iterator object
         * 
         * \param aPointer  Pointer to the input array data 
         * \param Stride    Stride in the array of data
         */
        iterator(T const *aPointer, std::size_t const Stride);

        /*!
         * \brief Destroy the iterator object
         * 
         */
        ~iterator();

        /*!
         * \brief Operator `==` compares the underlying iterators 
         * 
         * \param rhs  Iterator adaptor to compare 
         * 
         * \returns true If `lhs = rhs` 
         */
        inline bool operator==(iterator const &rhs);

        /*!
         * \brief Operator `!=` compares the underlying iterators 
         * 
         * \param rhs  Iterator adaptor to compare 
         * 
         * \returns true If `lhs != rhs` 
         */
        inline bool operator!=(iterator const &rhs);

        /*!
         * \brief Operator `<` compares the underlying iterators 
         * 
         * \param rhs  Iterator adaptor to compare 
         * 
         * \returns true If `lhs < rhs` 
         */
        inline bool operator<(iterator const &rhs);

        /*!
         * \brief Operator `<=` compares the underlying iterators 
         * 
         * \param rhs  Iterator adaptor to compare 
         * 
         * \returns true If `lhs <= rhs` 
         */
        inline bool operator<=(iterator const &rhs);

        /*!
         * \brief Operator `>` compares the underlying iterators 
         * 
         * \param rhs  Iterator adaptor to compare 
         * 
         * \returns true If `lhs > rhs` 
         */
        inline bool operator>(iterator const &rhs);

        /*!
         * \brief Operator `>=` compares the underlying iterators 
         * 
         * \param rhs  Iterator adaptor to compare 
         * 
         * \returns true If `lhs >= rhs` 
         */
        inline bool operator>=(iterator const &rhs);

        /*!
         * \brief Operator `++` advances the iterator 
         * 
         * \returns iterator& The incremented iterator
         */
        inline iterator &operator++();

        /*!
         * \brief Operator `++` advances the iterator 
         * 
         * \returns iterator The incremented iterator 
         */
        inline iterator operator++(int);

        /*!
         * \brief Operator `+` advances the iterator incremented by n. 
         * 
         * \param n  The number of positions to increment the iterator 
         * 
         * \returns iterator Returns an iterator which is advanced by n positions.
         */
        inline iterator operator+(int n) const;

        /*!
         * \brief Operator `+=` advances the iterator incremented by n. 
         * 
         * \param n  The number of positions to increment the iterator 
         * 
         * \returns iterator Advances the iterator by n positions.
         */
        inline iterator &operator+=(int n);

        /*!
         * \brief Access element at the current index 
         * 
         * \returns Actual value at the current index 
         */
        inline T &operator*();

        /*!
         * \brief Access element at the current index 
         * 
         * \returns Actual value at the current index 
         */
        inline T const operator*() const;

        /*! 
         * \brief Get a pointer to the managed object or nullptr if no object is owned
         * 
         * \returns a pointer to the managed object or nullptr if no object is owned 
         */
        inline T *get() const;

      public:
        //! Iterator category
        using iterator_category = std::forward_iterator_tag;

        //! Value type - The type "pointed to" by the iterator.
        using value_type = std::remove_cv<T>;

        //! Distance between iterators is represented as this type.
        using difference_type = std::ptrdiff_t;

        //! This type represents a pointer-to-value_type.
        using pointer = T *;

        //! This type represents a reference-to-value_type.
        using reference = T &;

      private:
        //! Iterator position
        T *iteratorPosition;

        //! Stride in the input array of data
        std::size_t inStride;
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
    arrayWrapper(T *InputArray, int const NumOfElements, int const Stride = 1);

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
     * \param NumOfElements Size of the array 
     * \param Stride        Stride in the array elements
     */
    arrayWrapper(std::unique_ptr<T[]> &InputArray, int const NumOfElements, int const Stride = 1);

    /*!
     * \brief Construct a new Array Wrapper object
     * 
     * \param InputArray    Input array data 
     * \param NumOfElements Size of the array 
     * \param Stride        Stride in the array elements
     */
    arrayWrapper(std::unique_ptr<T[]> const &InputArray, int const NumOfElements, int const Stride = 1);

    /*!
     * \brief Construct a new Array Wrapper object
     * 
     * \param InputArray  Input array data 
     * \param Stride      Stride in the array elements
     */
    arrayWrapper(std::vector<T> &InputArray, int const Stride = 1);

    /*!
     * \brief Construct a new Array Wrapper object
     * 
     * \param InputArray  Input array data 
     * \param Stride      Stride in the array elements
     */
    arrayWrapper(std::vector<T> const &InputArray, int const Stride = 1);

    /*!
     * \brief Move constructor Construct a new Array Wrapper object
     * 
     * \param other Array Wrapper object
     */
    arrayWrapper(arrayWrapper<T> &&other);

    /*!
     * \brief Move assignment 
     * 
     * \param other Array Wrapper object
     * 
     * \returns arrayWrapper& 
     */
    arrayWrapper<T> &operator=(arrayWrapper<T> &&other);

    /*!
     * \brief Destroy the Array Wrapper object
     * 
     */
    ~arrayWrapper(){};

    /*!
     * \brief Set the wrapper
     * 
     * \param InputArray    Input array data 
     * \param NumOfElements Size of the input array
     * \param Stride        Stride in the array elements
     */
    inline void set(T *InputArray, int const NumOfElements, int const Stride = 1);

    /*!
     * \brief Set the wrapper
     * 
     * \param InputArray    Input array data 
     * \param NumOfElements Size of the input array
     * \param Stride        Stride in the array elements
     */
    inline void set(T const *InputArray, int const NumOfElements, int const Stride = 1);

    /*! 
     * \brief Returns an iterator to the beginning of Input
     * 
     * \returns An iterator to the beginning of the given Input
     */
    inline iterator begin();

    /*! 
     * \brief Returns an iterator to the beginning of Input
     * 
     * \returns An iterator to the beginning of the given Input
     */
    inline iterator begin() const;

    /*! 
     * \brief Returns an iterator to the end
     * 
     * \returns An iterator to the end of the given Input
     */
    inline iterator end();

    /*! 
     * \brief Returns an iterator to the end
     * 
     * \returns An iterator to the end of the given Input
     */
    inline iterator end() const;

    /*!
     * \brief Get the size of array
     * 
     * \returns Size of the array
     */
    inline std::size_t size() const;

    /*!
     * \brief Get the Stride
     * 
     * \returns Stride of the array
     */
    inline std::size_t stride() const;

    /*!
     * \brief Swap with the input arraywrapper object
     * 
     * \param other arraywrapper object
     */
    inline void swap(arrayWrapper<T> &other);

    /*!
     * \brief Access element at provided index id with checking bounds
     * 
     * \param id  Requested index 
     * 
     * \returns Element at (id)
     */
    inline T at(int const id) const;

    /*!
     * \brief Access element at provided index
     * 
     * \param id  Requested id
     * 
     * \returns Element at (id)
     */
    inline T operator()(int const id) const;

    /*!
     * \brief Access element at provided index
     * 
     * \param id  Requested id
     * 
     * \returns Element at (id)
     */
    inline T operator[](int const id) const;

  protected:
    /*!
     * \brief Delete a arrayWrapper object copy construction
     * 
     * Make it noncopyable.
     */
    arrayWrapper(arrayWrapper const &) = delete;

    /*!
     * \brief Delete a arrayWrapper object assignment
     * 
     * Make it nonassignable
     * 
     * \returns arrayWrapper& 
     */
    arrayWrapper &operator=(const arrayWrapper &) = delete;

  private:
    //! Pointer to the actual Input with only read access
    T const *inArray;

    //! Pointer to the actual Input with read/write access
    T *inOutArray;

    //! Size of InputArray
    std::size_t inNumOfElements;

    //! Stride in the input array of data (default is 1)
    std::size_t inStride;
};

template <class T>
arrayWrapper<T>::arrayWrapper() : inArray(nullptr), inOutArray(nullptr), inNumOfElements(0), inStride(1) {}

template <class T>
arrayWrapper<T>::arrayWrapper(T *InputArray, int const NumOfElements, int const Stride) : inArray(nullptr),
                                                                                          inOutArray(InputArray),
                                                                                          inNumOfElements(NumOfElements),
                                                                                          inStride(Stride) {}

template <class T>
arrayWrapper<T>::arrayWrapper(T const *InputArray, int const NumOfElements, int const Stride) : inArray(InputArray),
                                                                                                inOutArray(nullptr),
                                                                                                inNumOfElements(NumOfElements),
                                                                                                inStride(Stride) {}

template <class T>
arrayWrapper<T>::arrayWrapper(std::unique_ptr<T[]> &InputArray, int const NumOfElements, int const Stride) : inArray(InputArray.get()),
                                                                                                             inOutArray(InputArray.get()),
                                                                                                             inNumOfElements(NumOfElements),
                                                                                                             inStride(Stride) {}

template <class T>
arrayWrapper<T>::arrayWrapper(std::unique_ptr<T[]> const &InputArray, int const NumOfElements, int const Stride) : inArray(InputArray.get()),
                                                                                                                   inOutArray(nullptr),
                                                                                                                   inNumOfElements(NumOfElements),
                                                                                                                   inStride(Stride) {}

template <class T>
arrayWrapper<T>::arrayWrapper(std::vector<T> &InputArray, int const Stride) : inArray(InputArray.data()),
                                                                              inOutArray(InputArray.data()),
                                                                              inNumOfElements(InputArray.size()),
                                                                              inStride(Stride) {}

template <class T>
arrayWrapper<T>::arrayWrapper(std::vector<T> const &InputArray, int const Stride) : inArray(InputArray.data()),
                                                                                    inOutArray(nullptr),
                                                                                    inNumOfElements(InputArray.size()),
                                                                                    inStride(Stride) {}

template <class T>
arrayWrapper<T>::arrayWrapper(arrayWrapper<T> &&other)
{
    inArray = std::move(other.inArray);
    inOutArray = std::move(other.inOutArray);
    inNumOfElements = other.inNumOfElements;
    inStride = other.inStride;
}

template <class T>
arrayWrapper<T> &arrayWrapper<T>::operator=(arrayWrapper<T> &&other)
{
    inArray = std::move(other.inArray);
    inOutArray = std::move(other.inOutArray);
    inNumOfElements = other.inNumOfElements;
    inStride = other.inStride;

    return *this;
}

template <class T>
inline void arrayWrapper<T>::set(T *InputArray, int const NumOfElements, int const Stride)
{
    inArray = nullptr;
    inOutArray = InputArray;
    inNumOfElements = (NumOfElements >= 0) ? static_cast<std::size_t>(NumOfElements) : throw(std::runtime_error("Wrong index!"));
    inStride = (Stride >= 0) ? static_cast<std::size_t>(Stride) : throw(std::runtime_error("Wrong index!"));
}

template <class T>
inline void arrayWrapper<T>::set(T const *InputArray, int const NumOfElements, int const Stride)
{
    inArray = InputArray;
    inOutArray = nullptr;
    inNumOfElements = (NumOfElements >= 0) ? static_cast<std::size_t>(NumOfElements) : throw(std::runtime_error("Wrong index!"));
    inStride = (Stride >= 0) ? static_cast<std::size_t>(Stride) : throw(std::runtime_error("Wrong index!"));
}

template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::begin()
{
    return inArray ? ((inStride == 1) ? arrayWrapper<T>::iterator(inArray) : arrayWrapper<T>::iterator(inArray, inStride)) : ((inStride == 1) ? arrayWrapper<T>::iterator(inOutArray) : arrayWrapper<T>::iterator(inOutArray, inStride));
}

template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::begin() const
{
    return inArray ? ((inStride == 1) ? arrayWrapper<T>::iterator(inArray) : arrayWrapper<T>::iterator(inArray, inStride)) : ((inStride == 1) ? arrayWrapper<T>::iterator(inOutArray) : arrayWrapper<T>::iterator(inOutArray, inStride));
}

template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::end()
{
    return inArray ? (arrayWrapper<T>::iterator(inArray + inNumOfElements)) : (arrayWrapper<T>::iterator(inOutArray + inNumOfElements));
}

template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::end() const
{
    return inArray ? (arrayWrapper<T>::iterator(inArray + inNumOfElements)) : (arrayWrapper<T>::iterator(inOutArray + inNumOfElements));
}

template <class T>
inline std::size_t arrayWrapper<T>::size() const { return inStride == 1 ? inNumOfElements : inNumOfElements / inStride; }

template <class T>
inline std::size_t arrayWrapper<T>::stride() const { return inStride; }

template <class T>
inline void arrayWrapper<T>::swap(arrayWrapper<T> &other)
{
    std::swap(inArray, other.inArray);
    std::swap(inOutArray, other.inOutArray);
    std::swap(inNumOfElements, other.inNumOfElements);
    std::swap(inStride, other.inStride);
}

template <class T>
inline T arrayWrapper<T>::at(int const id) const
{
    std::size_t const i = id * inStride;
    return (i < inNumOfElements) ? (inArray ? inArray[i] : inOutArray[i]) : throw(std::runtime_error("Index out of bound!"));
}

template <class T>
inline T arrayWrapper<T>::operator()(int const id) const
{
    return inArray ? inArray[id * inStride] : inOutArray[id * inStride];
}

template <class T>
inline T arrayWrapper<T>::operator[](int const id) const
{
    return inArray ? inArray[id * inStride] : inOutArray[id * inStride];
}

template <class T>
arrayWrapper<T>::iterator::iterator() : iteratorPosition(nullptr), inStride(1) {}

template <class T>
arrayWrapper<T>::iterator::iterator(std::size_t const Stride) : iteratorPosition(nullptr), inStride(Stride) {}

template <class T>
arrayWrapper<T>::iterator::iterator(T *aPointer) : iteratorPosition(aPointer), inStride(1) {}

template <class T>
arrayWrapper<T>::iterator::iterator(T const *aPointer) : iteratorPosition(const_cast<T *>(aPointer)), inStride(1) {}

template <class T>
arrayWrapper<T>::iterator::iterator(T *aPointer, std::size_t const Stride) : iteratorPosition(aPointer), inStride(Stride) {}

template <class T>
arrayWrapper<T>::iterator::iterator(T const *aPointer, std::size_t const Stride) : iteratorPosition(const_cast<T *>(aPointer)), inStride(Stride) {}

template <class T>
arrayWrapper<T>::iterator::~iterator() {}

template <class T>
inline bool arrayWrapper<T>::iterator::operator==(arrayWrapper<T>::iterator const &rhs) { return iteratorPosition == rhs.iteratorPosition; }

template <class T>
inline bool arrayWrapper<T>::iterator::operator!=(arrayWrapper<T>::iterator const &rhs) { return inStride == 1 ? iteratorPosition != rhs.iteratorPosition : iteratorPosition < rhs.iteratorPosition; }

template <class T>
inline bool arrayWrapper<T>::iterator::operator<(arrayWrapper<T>::iterator const &rhs) { return iteratorPosition < rhs.iteratorPosition; }

template <class T>
inline bool arrayWrapper<T>::iterator::operator<=(arrayWrapper<T>::iterator const &rhs) { return iteratorPosition <= rhs.iteratorPosition; }

template <class T>
inline bool arrayWrapper<T>::iterator::operator>(arrayWrapper<T>::iterator const &rhs) { return iteratorPosition > rhs.iteratorPosition; }

template <class T>
inline bool arrayWrapper<T>::iterator::operator>=(arrayWrapper<T>::iterator const &rhs) { return iteratorPosition >= rhs.iteratorPosition; }

template <class T>
inline typename arrayWrapper<T>::iterator &arrayWrapper<T>::iterator::operator++()
{
    iteratorPosition += inStride;
    return *this;
}

template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::iterator::operator++(int)
{
    arrayWrapper<T>::iterator t(*this);
    operator++();
    return t;
}

template <class T>
inline typename arrayWrapper<T>::iterator arrayWrapper<T>::iterator::operator+(int n) const
{
    arrayWrapper<T>::iterator t(*this);
    for (int i = 0; i < n; i++)
    {
        t++;
    }
    return t;
}

template <class T>
inline typename arrayWrapper<T>::iterator &arrayWrapper<T>::iterator::operator+=(int n)
{
    for (int i = 0; i < n; i++)
    {
        iteratorPosition += inStride;
    }
    return *this;
}

template <class T>
inline T &arrayWrapper<T>::iterator::operator*()
{
    return *iteratorPosition;
}

template <class T>
inline T const arrayWrapper<T>::iterator::operator*() const
{
    return *iteratorPosition;
}

template <class T>
inline T *arrayWrapper<T>::iterator::get() const
{
    return iteratorPosition;
}

} // namespace umuq

#endif // UMUQ_ARRAYWRAPPER
