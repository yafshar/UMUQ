#ifndef UMHBM_ARRAY_H
#define UMHBM_ARRAY_H

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
        iterator() : iPosition(nullptr) {}
        iterator(T const *aPointer) : iPosition(aPointer) {}
        ~iterator(){}; // nothing to do

        bool operator==(iterator const &rhs) { return iPosition == rhs.iPosition; }
        bool operator!=(iterator const &rhs) { return iPosition != rhs.iPosition; }

        iterator &operator++()
        {
            ++iPosition;
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
    };

    /*! 
     *  \brief ArrayWrapper constructor
     *
     */
    ArrayWrapper() : iArray(nullptr), iNumOfElements(0) {}
    ArrayWrapper(T const *InputArray, size_t NumOfElements) : iArray(InputArray), iNumOfElements(NumOfElements) {}
    ArrayWrapper(ArrayWrapper &&InputArrayObj)
    {
        iArray = InputArrayObj.iArray;
        InputArrayObj.iArray = nullptr;
        iNumOfElements = InputArrayObj.iNumOfElements;
        InputArrayObj.iNumOfElements = 0;
    }
    ArrayWrapper &operator=(ArrayWrapper &&InputArrayObj)
    {
        iArray = InputArrayObj.iArray;
        InputArrayObj.iArray = nullptr;
        iNumOfElements = InputArrayObj.iNumOfElements;
        InputArrayObj.iNumOfElements = 0;
        return *this;
    }

    ~ArrayWrapper(){};

    inline void set(T *InputArray, size_t NumOfElements)
    {
        iArray = InputArray;
        iNumOfElements = NumOfElements;
    }

    /*! 
     * \brief Returns an iterator to the beginning of Input
     * 
     * \return an iterator to the beginning of the given Input
     */
    inline iterator begin()
    {
        return iterator(iArray);
    }
    inline const iterator begin() const
    {
        return iterator(iArray);
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
    inline const iterator end() const
    {
        return iterator(iArray + iNumOfElements);
    }

    inline size_t size() const { return iNumOfElements; }

    inline void swap(ArrayWrapper<T> &aObj)
    {
        std::swap(iNumOfElements, aObj.iNumOfElements);
        std::swap(iArray, aObj.iArray);
    }

  private:
    //! Pointer to the actual Input
    T const *iArray;
    
    //! Size of InputArray
    size_t iNumOfElements;

    // make it noncopyable
    ArrayWrapper(const ArrayWrapper &) = delete;

    // make it not assignable
    ArrayWrapper &operator=(const ArrayWrapper &) = delete;
};

#endif