#ifndef UMHBM_ARRAY_H
#define UMHBM_ARRAY_H

#include <iostream>

/*! \class ArrayWrapper
*   \brief ArrayWrapper is a class which creates a std iterator for a vector
*	
*/
template <typename T>
class ArrayWrapper
{
    class iterator : public std::iterator<std::input_iterator_tag, T>
    {
      public:
        iterator(const T *aPointer) : iPosition(aPointer) {}

        bool operator==(const iterator &rhs) { return iPosition == rhs.iPosition; }
        bool operator!=(const iterator &rhs) { return iPosition != rhs.iPosition; }

        void operator++() { ++iPosition; }

        T operator*()
        {
            return *iPosition;
        }

        ~iterator()
        {
            // nothing to do
        }

      private:
        const T *iPosition;
    };

  public:
    ArrayWrapper(const T *aInputArray, long aNumOfElements) : iArray(aInputArray), iNumOfElements(aNumOfElements) {}

    iterator begin()
    {
        return iterator(iArray);
    }
    iterator end()
    {
        return iterator(iArray + iNumOfElements);
    }

  private:
    const T *iArray;
    long iNumOfElements;
};

#endif