#ifndef UMHBM_ARRAY_H
#define UMHBM_ARRAY_H

#include <iostream>
#include <iterator>

/*! \class ArrayWrapper
*   \brief ArrayWrapper is a class which creates a std iterator for an array of type T
*	
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
    ArrayWrapper(T const *InputT, long NumOfElements) : iT(InputT), iNumOfElements(NumOfElements) {}
    ~ArrayWrapper(){};

    /*! 
     * \brief Returns an iterator to the beginning of Input
     * 
     * \return an iterator to the beginning of the given Input
     */
    iterator begin()
    {
        return iterator(iT);
    }
    /*! 
     * \brief Returns an iterator to the end
     * 
     * \return an iterator to the end of the given Input
     */
    iterator end()
    {
        return iterator(iT + iNumOfElements);
    }

  private:
    //! Pointer to the actual Input
    T const *iT;
    //! Size of InputT
    long iNumOfElements;
};

#endif