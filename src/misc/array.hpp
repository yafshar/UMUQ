#ifndef UMUQ_ARRAY_H
#define UMUQ_ARRAY_H

#include <iostream>
#include <iterator>

/*! \class ArrayWrapper
  * \brief ArrayWrapper is a class which creates a std iterator for an array of type T
  *
  * Expression of a class T (vector or matrix, or other types) as an array object
  */
template <class T>
class ArrayWrapper
{
  public:
    class iterator : public std::iterator<std::input_iterator_tag, T>
    {
      public:
        /*! 
         *  \brief iterator constructor
         *
         */
        iterator() : iPosition(nullptr), iStride(1) {}
        iterator(std::size_t Stride) : iPosition(nullptr), iStride(Stride) {}
        iterator(T const *aPointer) : iPosition(aPointer), iStride(1) {}
        iterator(T const *aPointer, std::size_t Stride) : iPosition(aPointer), iStride(Stride) {}

        ~iterator(){}; // nothing to do

        bool operator==(iterator const &rhs) { return iPosition == rhs.iPosition; }
        bool operator!=(iterator const &rhs) { return iStride == 1 ? iPosition != rhs.iPosition : iPosition < rhs.iPosition; }

        iterator &operator++()
        {
            iPosition += iStride;
            return *this;
        }

        iterator operator++(int)
        {
            iterator t(*this);
            operator++();
            return t;
        }

        T operator*()
        {
            return *iPosition;
        }

        /*! 
         *  \brief Get the actual value
         * 
         * \return the actual value
         */
        inline T get()
        {
            return *iPosition;
        }

      private:
        T const *iPosition;
        std::size_t iStride;
    };

    /*! 
     *  \brief ArrayWrapper constructor
     *
     */
    ArrayWrapper() : iArray(nullptr), iNumOfElements(0), iStride(1) {}
    ArrayWrapper(T const *InputArray, std::size_t NumOfElements, std::size_t Stride = 1) : iArray(InputArray),
                                                                                           iNumOfElements(NumOfElements),
                                                                                           iStride(Stride) {}

    ArrayWrapper(std::unique_ptr<T[]> const &InputArray, std::size_t NumOfElements, std::size_t Stride = 1) : iArray(InputArray.get()),
                                                                                                              iNumOfElements(NumOfElements),
                                                                                                              iStride(Stride) {}

    /*!
     * \brief Move constructor Construct a new Array Wrapper object
     * 
     * \param InputArrayObj 
     */
    ArrayWrapper(ArrayWrapper &&InputArrayObj)
    {
        iArray = InputArrayObj.iArray;
        iNumOfElements = InputArrayObj.iNumOfElements;
        iStride = InputArrayObj.iStride;

        InputArrayObj.iArray = nullptr;
        InputArrayObj.iNumOfElements = 0;
        InputArrayObj.iStride = 1;
    }

    /*!
     * \brief Move assignment 
     * 
     * \param InputArrayObj 
     * \return ArrayWrapper& 
     */
    ArrayWrapper &operator=(ArrayWrapper &&InputArrayObj)
    {
        iArray = InputArrayObj.iArray;
        iNumOfElements = InputArrayObj.iNumOfElements;
        iStride = InputArrayObj.iStride;

        InputArrayObj.iArray = nullptr;
        InputArrayObj.iNumOfElements = 0;
        InputArrayObj.iStride = 1;

        return *this;
    }

    /*!
     * \brief Destroy the Array Wrapper object
     * 
     */
    ~ArrayWrapper(){};

    /*!
     * \brief set the wrapper
     * 
     * \param InputArray 
     * \param NumOfElements 
     */
    inline void set(T *InputArray, std::size_t NumOfElements, std::size_t Stride = 1)
    {
        iArray = InputArray;
        iNumOfElements = NumOfElements;
        iStride = Stride;
    }

    /*! 
     * \brief Returns an iterator to the beginning of Input
     * 
     * \return an iterator to the beginning of the given Input
     */
    inline iterator begin()
    {
        return iStride == 1 ? iterator(iArray) : iterator(iArray, iStride);
    }

    inline iterator begin() const
    {
        return iStride == 1 ? iterator(iArray) : iterator(iArray, iStride);
    }

    /*! 
     * \brief Returns an iterator to the end
     * 
     * \return an iterator to the end of the given Input
     */
    inline iterator end()
    {
        return iterator(iArray + iNumOfElements);
    }

    inline iterator end() const
    {
        return iterator(iArray + iNumOfElements);
    }

    /*!
     * \brief Get the size of array
     * 
     * \return Size of the array
     */
    inline std::size_t size() const { return iStride == 1 ? iNumOfElements : iNumOfElements / iStride; }

    /*!
     * \brief Swap with the input arraywrapper object
     * 
     * \param InputArrayWrapperObj 
     */
    inline void swap(ArrayWrapper<T> &InputArrayWrapperObj)
    {
        std::swap(iNumOfElements, InputArrayWrapperObj.iNumOfElements);
        std::swap(iStride, InputArrayWrapperObj.iStride);
        std::swap(iArray, InputArrayWrapperObj.iArray);
    }

  private:
    //! Pointer to the actual Input
    T const *iArray;

    //! Size of InputArray
    std::size_t iNumOfElements;

    //! stride
    std::size_t iStride;

    // make it noncopyable
    ArrayWrapper(const ArrayWrapper &) = delete;

    // make it not assignable
    ArrayWrapper &operator=(const ArrayWrapper &) = delete;
};

#endif