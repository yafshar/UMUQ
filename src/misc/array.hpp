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
        iterator(T const *aPointer) : iPosition(aPointer) {}

        bool operator==(iterator const &rhs) { return iPosition == rhs.iPosition; }
        bool operator!=(iterator const &rhs) { return iPosition != rhs.iPosition; }

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
        T const *iPosition;
    };

  public:
    ArrayWrapper(T const *aInputArray, long aNumOfElements) : iArray(aInputArray), iNumOfElements(aNumOfElements) {}

    iterator begin()
    {
        return iterator(iArray);
    }
    iterator end()
    {
        return iterator(iArray + iNumOfElements);
    }

  private:
    T const *iArray;
    long iNumOfElements;
};

#endif