#ifndef UMUQ_DATATYPE_H
#define UMUQ_DATATYPE_H

#include <iostream>
#include "../io/io.hpp"

/*!
* \brief structure for sorting Fvalue for entires of database structure
* 
* \tparam T data type
*
* \param idx      an intger argument for indexing
* \param nsel     an integer argument for selection of leaders only
* \param Fvalue   a double argument for function value
*/
template <typename T>
struct sort_t
{
    int idx;
    int nsel;
    T Fvalue;
};

/*!
 * \brief database structure
 *
 * \tparam T data type
 * \tparam T type of database structure
 * 
 * \param entry
 * \param entries an integer argument shows the size of entry
 * \param m A mutex object
 */
template <class DBT>
class database
{
  public:
    /*!
     *  \brief constructor for the database structure
     *    
     */
    database() : entry(nullptr),
                 entries(0)
    {
        pthread_mutex_init(&m, nullptr);
    }

    /*!
     * /brief Init function taking two arguments and initialize the structure.
     *
     *  \param nsize1 an integer argument.
     */
    bool init(int nsize1);

    /*!
     * /brief A member updating the database
     *
     *  \param Parray     a double array of points.
     *  \param ndimParray an integer argument, shows the size of Parray
     *  \param Fvalue     a double value 
     *  \param Garray     a double array
     *  \param ndimGarray an integer argument, shows the size of Garray
     *  \param surrogate  an integer argument for the surrogate model (default 0, no surrogate)
     */
    bool update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray, int surrogate = 0);

    /*!
     * /brief function for sorting elemnts of an array for database elements.
     *
     *  Sorts the entries elements of the array pointed to by list, each 
     *  element size bytes long, using the compar function to determine the order.
     */
    inline void sort(sort_t<double> *list);

    /*!
     * /brief function for printing  the data
     *
     */
    bool print()
    {
        if (entry != nullptr)
        {
            std::cout << "---- database priniting ----" << std::endl;

            for (int pos = 0; pos < entries; pos++)
            {
                if (entry[pos].Parray != nullptr)
                {
                    int j;
                    std::cout << "ENTRY"
                              << std::setw(5) << pos << " : POINT(";
                    for (j = 0; j < entry[pos].ndimParray - 1; j++)
                    {
                        std::cout << std::setw(20) << entry[pos].Parray[j] << ", ";
                    }
                    std::cout << std::setw(20) << entry[pos].Parray[j] << ") Fvalue="
                              << std::setw(20) << entry[pos].Fvalue << " Surrogate="
                              << std::setw(20) << entry[pos].surrogate << std::endl;
                }
                if (entry[pos].Garray != nullptr)
                {
                    int i;
                    std::cout << "Garray=[";
                    for (i = 0; i < entry[pos].ndimGarray - 1; i++)
                    {
                        std::cout << std::setw(20) << entry[pos].Garray[i] << ", ";
                    }
                    std::cout << std::setw(20) << entry[pos].Garray[i] << "]" << std::endl;
                }
            }

            std::cout << "----------------------------" << std::endl;

            return true;
        }
        return false;
    }

    /*!
     * /brief function for dumping the data
     *
     */
    bool dump(const char *fname = "")
    {
        if (entry != nullptr)
        {
            char fileName[256];
            if (strlen(fname) == 0)
            {
                sprintf(fileName, "db_%03d.txt", entries - 1);
            }
            else
            {
                sprintf(fileName, "%s_%03d.txt", fname, entries - 1);
            }

            io f;
            if (f.openFile(fileName, f.out | f.trunc))
            {
                double **tmp = nullptr;
                int nRows = 2 + (int)(entry[0].Garray != nullptr);
                tmp = new double *[3];

                for (int pos = 0; pos < entries - 1; pos++)
                {
                    tmp[0] = entry[pos].Parray;
                    tmp[1] = &entry[pos].Fvalue;
                    tmp[2] = entry[pos].Garray;

                    int nCols = entry[pos].ndimParray + 1 + entry[pos].ndimGarray;

                    if (!f.saveMatrix<double>(tmp, nRows, nCols, 2))
                    {
                        return false;
                    }
                }

                delete[] tmp;

                f.closeFile();

                return true;
            }
            return false;
        }
        return false;
    }

    /*!
     * /brief function for loading the data
     *
     */
    bool load(const char *fname)
    {
        if (entry != nullptr)
        {
            char fileName[256];
            if (strlen(fname) == 0)
            {
                sprintf(fileName, "db_%03d.txt", entries - 1);
            }
            else
            {
                sprintf(fileName, "%s_%03d.txt", fname, entries - 1);
            }

            io f;
            if (f.openFile(fileName, f.in))
            {
                double **tmp = nullptr;

                int nRows = 2 + (int)(entry[0].Garray != nullptr);
                tmp = new double *[3];

                for (int pos = 0; pos < entries - 1; pos++)
                {
                    tmp[0] = entry[pos].Parray;
                    tmp[1] = &entry[pos].Fvalue;
                    tmp[2] = entry[pos].Garray;

                    int nCols = entry[pos].ndimParray + 1;

                    if (!f.loadMatrix<double>(tmp, nRows, nCols, 1))
                    {
                        return false;
                    }
                }

                delete[] tmp;

                f.closeFile();

                return true;
            }
            return false;
        }
        return false;
    }

  public:
    //! Data container
    DBT *entry;

    //! Number of entries
    int entries;

  private:
  
    //! Mutex object
    pthread_mutex_t m;
};

template <class T>
bool database<T>::init(int nsize1)
{
    if (entry == nullptr)
    {
        try
        {
            entry = new T[nsize1];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        };

        for (int i = 0; i < nsize1; i++)
        {
            entry[i] = (T)0;
        }
        return true;
    }
    return false;
}

template <class T>
bool database<T>::update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray, int surrogate)
{
    int pos;

    pthread_mutex_lock(&m);
    pos = entries;
    entries++;
    pthread_mutex_unlock(&m);

    if (ndimParray > entry[pos].ndimParray)
    {
        if (entry[pos].Parray != nullptr)
        {
            delete[] entry[pos].Parray;
        }

        try
        {
            entry[pos].Parray = new double[ndimParray];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }
    }
    entry[pos].ndimParray = ndimParray;

    for (int i = 0; i < ndimParray; i++)
    {
        entry[pos].Parray[i] = Parray[i];
    }

    entry[pos].Fvalue = Fvalue;

    if (ndimGarray > entry[pos].ndimGarray)
    {
        if (entry[pos].Garray != nullptr)
        {
            delete[] entry[pos].Garray;
        }

        try
        {
            entry[pos].Garray = new double[ndimGarray];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }
    }
    entry[pos].ndimGarray = ndimGarray;

    for (int i = 0; i < ndimGarray; i++)
    {
        entry[pos].Garray[i] = Garray[i];
    }

    entry[pos].surrogate = surrogate;
    return true;
}

/*!
 * \brief Pointer to a function that compares two elements.
 *   
 *  This function is called repeatedly by qsort to compare two elements.
 */
int compar_desc(const void *p1, const void *p2)
{
    const sort_t<double> *s1 = static_cast<const sort_t<double> *>(p1);
    const sort_t<double> *s2 = static_cast<const sort_t<double> *>(p2);

    /* -: ascending order, +: descending order */
    return (s2->nsel - s1->nsel);
}

/*!
 * \brief quick sort
 * 
 * \tparam T 
 * 
 * \param list array which we want to sort it 
 */
template <class T>
inline void database<T>::sort(sort_t<double> *list)
{
    qsort(list, entries, sizeof(sort_t<double>), compar_desc);
}

template<typename T>
struct cgdb_t : public database<cgdbp_t<T>>
{
};

template<typename T>
struct db_t : public database<dbp_t<T>>
{
};

template<typename T>
struct resdb_t : public database<resdbp_t<T>>
{
};

#endif
