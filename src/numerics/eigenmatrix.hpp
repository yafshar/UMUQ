#ifndef UMHBM_EIGENMATRIX_H
#define UMHBM_EIGENMATRIX_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <Eigen/Dense>

// Standard typedef from eigen
typedef typename Eigen::Matrix<double, 2, 2> EMatrix2d;
typedef typename Eigen::Matrix<double, 2, Eigen::Dynamic> EMatrix2Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 2> EMatrixX2d;

typedef typename Eigen::Matrix<double, 3, 3> EMatrix3d;
typedef typename Eigen::Matrix<double, 3, Eigen::Dynamic> EMatrix3Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 3> EMatrixX3d;

typedef typename Eigen::Matrix<double, 4, 4> EMatrix4d;
typedef typename Eigen::Matrix<double, 4, Eigen::Dynamic> EMatrix4Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 4> EMatrixX4d;

typedef typename Eigen::Matrix<double, 5, 5> EMatrix5d;
typedef typename Eigen::Matrix<double, 5, Eigen::Dynamic> EMatrix5Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 5> EMatrixX5d;

typedef typename Eigen::Matrix<double, 6, 6> EMatrix6d;
typedef typename Eigen::Matrix<double, 6, Eigen::Dynamic> EMatrix6Xd;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 6> EMatrixX6d;

typedef typename Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EMatrixXd;

typedef typename Eigen::Matrix<float, 2, 2> EMatrix2f;
typedef typename Eigen::Matrix<float, 2, Eigen::Dynamic> EMatrix2Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 2> EMatrixX2f;

typedef typename Eigen::Matrix<float, 3, 3> EMatrix3f;
typedef typename Eigen::Matrix<float, 3, Eigen::Dynamic> EMatrix3Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 3> EMatrixX3f;

typedef typename Eigen::Matrix<float, 4, 4> EMatrix4f;
typedef typename Eigen::Matrix<float, 4, Eigen::Dynamic> EMatrix4Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 4> EMatrixX4f;

typedef typename Eigen::Matrix<float, 5, 5> EMatrix5f;
typedef typename Eigen::Matrix<float, 5, Eigen::Dynamic> EMatrix5Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 5> EMatrixX5f;

typedef typename Eigen::Matrix<float, 6, 6> EMatrix6f;
typedef typename Eigen::Matrix<float, 6, Eigen::Dynamic> EMatrix6Xf;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 6> EMatrixX6f;

typedef typename Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> EMatrixXf;

typedef typename Eigen::Matrix<int, 2, 2> EMatrix2i;
typedef typename Eigen::Matrix<int, 2, Eigen::Dynamic> EMatrix2Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 2> EMatrixX2i;

typedef typename Eigen::Matrix<int, 3, 3> EMatrix3i;
typedef typename Eigen::Matrix<int, 3, Eigen::Dynamic> EMatrix3Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 3> EMatrixX3i;

typedef typename Eigen::Matrix<int, 4, 4> EMatrix4i;
typedef typename Eigen::Matrix<int, 4, Eigen::Dynamic> EMatrix4Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 4> EMatrixX4i;

typedef typename Eigen::Matrix<int, 5, 5> EMatrix5i;
typedef typename Eigen::Matrix<int, 5, Eigen::Dynamic> EMatrix5Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 5> EMatrixX5i;

typedef typename Eigen::Matrix<int, 6, 6> EMatrix6i;
typedef typename Eigen::Matrix<int, 6, Eigen::Dynamic> EMatrix6Xi;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 6> EMatrixX6i;

typedef typename Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> EMatrixXi;

typedef typename Eigen::Matrix<double, 1, 2> ERowVector2d;
typedef typename Eigen::Matrix<double, 1, 3> ERowVector3d;
typedef typename Eigen::Matrix<double, 1, 4> ERowVector4d;
typedef typename Eigen::Matrix<double, 1, 5> ERowVector5d;
typedef typename Eigen::Matrix<double, 1, 6> ERowVector6d;
typedef typename Eigen::Matrix<double, 1, Eigen::Dynamic> ERowVectorXd;

typedef typename Eigen::Matrix<float, 1, 2> ERowVector2f;
typedef typename Eigen::Matrix<float, 1, 3> ERowVector3f;
typedef typename Eigen::Matrix<float, 1, 4> ERowVector4f;
typedef typename Eigen::Matrix<float, 1, 5> ERowVector5f;
typedef typename Eigen::Matrix<float, 1, 6> ERowVector6f;
typedef typename Eigen::Matrix<float, 1, Eigen::Dynamic> ERowVectorXf;

typedef typename Eigen::Matrix<int, 1, 2> ERowVector2i;
typedef typename Eigen::Matrix<int, 1, 3> ERowVector3i;
typedef typename Eigen::Matrix<int, 1, 4> ERowVector4i;
typedef typename Eigen::Matrix<int, 1, 5> ERowVector5i;
typedef typename Eigen::Matrix<int, 1, 6> ERowVector6i;
typedef typename Eigen::Matrix<int, 1, Eigen::Dynamic> ERowVectorXi;

typedef typename Eigen::Matrix<double, 2, 1> EVector2d;
typedef typename Eigen::Matrix<double, 3, 1> EVector3d;
typedef typename Eigen::Matrix<double, 4, 1> EVector4d;
typedef typename Eigen::Matrix<double, 5, 1> EVector5d;
typedef typename Eigen::Matrix<double, 6, 1> EVector6d;
typedef typename Eigen::Matrix<double, Eigen::Dynamic, 1> EVectorXd;

typedef typename Eigen::Matrix<float, 2, 1> EVector2f;
typedef typename Eigen::Matrix<float, 3, 1> EVector3f;
typedef typename Eigen::Matrix<float, 4, 1> EVector4f;
typedef typename Eigen::Matrix<float, 5, 1> EVector5f;
typedef typename Eigen::Matrix<float, 6, 1> EVector6f;
typedef typename Eigen::Matrix<float, Eigen::Dynamic, 1> EVectorXf;

typedef typename Eigen::Matrix<int, 2, 1> EVector2i;
typedef typename Eigen::Matrix<int, 3, 1> EVector3i;
typedef typename Eigen::Matrix<int, 4, 1> EVector4i;
typedef typename Eigen::Matrix<int, 5, 1> EVector5i;
typedef typename Eigen::Matrix<int, 6, 1> EVector6i;
typedef typename Eigen::Matrix<int, Eigen::Dynamic, 1> EVectorXi;

/*!
 * \brief Stores a set of parameters controlling the way matrices are printed
 * 
 * - precision \c FullPrecision.
 * - coeffSeparator string printed between two coefficients of the same row
 * - rowSeparator string printed between two rows
 */
Eigen::IOFormat fmt(Eigen::FullPrecision);

/*! 
 * \brief The Index type
 */
typedef typename std::ptrdiff_t Index;
std::ptrdiff_t width;

/*!
 * \brief New type to map the existing C++ memory buffer to an Eigen Matrix object in a RowMajor
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam TdataPtr typedef of the data pointer    
 * 
 */
template <typename TdataPtr>
using TEMapX = Eigen::Map<Eigen::Matrix<TdataPtr, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/*!
 * \brief New a read-only map type to map the existing C++ memory buffer to an Eigen Matrix object in a RowMajor
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam TdataPtr typedef of the data pointer    
 * 
 */
template <typename TdataPtr>
using CTEMapX = Eigen::Map<const Eigen::Matrix<TdataPtr, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/*!
 * \brief New type to map the existing C++ memory buffer to an Eigen Matrix object of type double in a RowMajor
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures.  
 * 
 */
using TEMapXd = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/*!
 * \brief New a read-only map type to map the existing C++ memory buffer to an Eigen Matrix object of type double in a RowMajor
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures.  
 * 
 */
using CTEMapXd = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/*!
 * \brief EMapX function copies the existing C++ memory buffer to an Eigen Matrix object of size(nRows, nCols) 
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures. 
 *  
 * \tparam TEMX     typedef for Eigen matrix 
 * \tparam TdataPtr typedef of the pointer to the array to map 
 * 
 * \param  dataPtr  pointer to the array to map of type TdataPtr
 * \param  nRows    Number of Rows in Matrix representation of Input array
 * \param  nCols    Number of Columns in Matrix representation of Input array
 * \param  EMapX    Eigen Matrix representation of the array    
 * 
 */
template <typename TEMX, typename TdataPtr>
TEMX EMapX(TdataPtr *dataPtr, size_t nRows, size_t nCols)
{
    return Eigen::Map<const TEMX>(dataPtr, nRows, nCols);
}

template <typename TEMX, typename TdataPtr>
TEMX EMapX(TdataPtr **dataPtr, size_t nRows, size_t nCols)
{
    TEMX MTemp(nRows, nCols);

    for (size_t i = 0; i < nRows; i++)
    {
        MTemp.row(i) = Eigen::Matrix<TdataPtr, Eigen::Dynamic, 1>::Map(&dataPtr[i][0], nCols);
    }

    return MTemp;
}

/*!
 * \brief Pointer will now point to a beginning of a memory buffer of the Eigen’s data structures
 *  
 * The Map operation maps the existing Eigen’s data structure to the memory buffer
 * 
 * \tparam TEMX     typedef for Eigen matrix 
 * \tparam TdataPtr typedef of the pointer to the array
 * 
 * \param  EMX     Input Eigen’s matrix of type TEMX
 * \param  dataPtr pointer to the array of type TdataPtr
 */
template <typename TEMX, typename TdataPtr>
void EMapX(const TEMX EMX, TdataPtr *dataPtr)
{
    Eigen::Map<TEMX>(dataPtr, EMX.rows(), EMX.cols()) = EMX;
}

template <typename TEMX, typename TdataPtr>
void EMapX(const TEMX EMX, TdataPtr **dataPtr)
{
    for (size_t i = 0; i < EMX.rows(); i++)
    {
        Eigen::Matrix<TdataPtr, Eigen::Dynamic, 1>::Map(&dataPtr[i][0], EMX.cols()) = EMX.row(i);
    }
}

/*!
 * \brief Copy the existing pointer to the C++ memory buffer of type double to an Eigen object
 * 
 * The Map operation maps the existing memory region into the Eigen’s data structures.  
 * 
 * \param  dataPtr  pointer to the array to map of type double
 * \param  nRows    Number of Rows
 * \param  nCols    Number of Columns
 * \param  EMapXd   Eigen Matrix representation of data
 */
EMatrixXd EMapXd(double *dataPtr, size_t nRows, size_t nCols)
{
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dataPtr, nRows, nCols);
}

EMatrixXd EMapXd(double **dataPtr, size_t nRows, size_t nCols)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MTemp(nRows, nCols);

    for (size_t i = 0; i < nRows; i++)
    {
        MTemp.row(i) = EVectorXd::Map(&dataPtr[i][0], nCols);
    }

    return MTemp;
}

/*!
 * \brief Pointer will now point to a beginning of a memory region of the Eigen’s data structures
 *  
 * The Map operation maps the existing Eigen’s data structure to the memory buffer
 * 
 * \param  EMXd    Input Eigen’s matrix of type double
 * \param  dataPtr Pointer to the memory buffer of type double
 */
void EMapXd(const EMatrixXd EMXd, double *dataPtr)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dataPtr, EMXd.rows(), EMXd.cols()) = EMXd;
}

void EMapXd(const EMatrixXd EMXd, double **dataPtr)
{
    for (size_t i = 0; i < EMXd.rows(); i++)
    {
        EVectorXd::Map(&dataPtr[i][0], EMXd.cols()) = EMXd.row(i);
    }
}

/*!
 * \brief Helper function to print the matrix
 * 
 * \tparam  Tidata type of data
 * 
 * \param   title  string that should be written at the top 
 * \param   idata  array of input data of type Tidata
 * \param   nRows  number of rows
 * \param   nCols  number of columns
 */
template <typename Tidata>
void printMatrix(const char *title, Tidata **idata, size_t nRows, size_t nCols)
{
    std::string sep = "\n----------------------------------------\n";
    std::cout << sep;
    std::cout << title << "\n\n";
    std::cout << EMapX<Eigen::Matrix<Tidata, Eigen::Dynamic, Eigen::Dynamic>, Tidata>(idata, nRows, nCols) << sep;
}

template <typename Tidata>
void printMatrix(Tidata **idata, size_t nRows, size_t nCols)
{
    std::string sep = "\n----------------------------------------\n";
    std::cout << sep;
    std::cout << EMapX<Eigen::Matrix<Tidata, Eigen::Dynamic, Eigen::Dynamic>, Tidata>(idata, nRows, nCols) << sep;
}

template <typename Tidata>
void printMatrix(const char *title, Tidata *idata, size_t nRows, size_t nCols)
{
    TEMapX<Tidata> TiMatrix(idata, nRows, nCols);

    std::string sep = "\n----------------------------------------\n";
    std::cout << sep;
    std::cout << title << "\n\n";
    std::cout << TiMatrix << sep;
}

template <typename Tidata>
void printMatrix(Tidata *idata, size_t nRows, size_t nCols)
{
    TEMapX<Tidata> TiMatrix(idata, nRows, nCols);

    std::string sep = "\n----------------------------------------\n";
    std::cout << sep;
    std::cout << TiMatrix << sep;
}

/*!
 * \brief Helper function to load the matrix of type TEMX from a file 
 * 
 * \tparam  TEMX   typedef for Eigen matrix 
 * \param   EMX    Eigen matrix
 */
template <typename TEMX>
inline bool loadMatrix(std::fstream &fs, TEMX &EMX)
{
    std::string line;

    for (int i = 0; i < EMX.rows(); i++)
    {
        if (getline(fs, line))
        {
            std::stringstream input_line(line);

            for (int j = 0; j < EMX.cols(); j++)
            {
                input_line >> EMX(i, j);
            }
        }
        else
        {
            return false;
        }
    }
    return true;
}

/*!
 * \brief Helper function to load the matrix from a file 
 * 
 * \tparam  Tidata data type 
 * \param   idata  array of input data of type Tidata
 * \param   nRows  number of rows
 * \param   nCols  number of columns
 * \param options  (default) 0 load matrix from matrix format and 1 load matrix from vector format
 */
template <typename Tidata>
inline bool loadMatrix(std::fstream &fs, Tidata **idata, size_t nRows, size_t nCols, size_t options = 0)
{
    std::string line;

    if (options == 0)
    {
        for (int i = 0; i < nRows; i++)
        {
            if (getline(fs, line))
            {
                std::stringstream input_line(line);

                for (int j = 0; j < nCols; j++)
                {
                    input_line >> idata[i][j];
                }
            }
            else
            {
                return false;
            }
        }
        return true;
    }
    else if (options == 1)
    {
        if (getline(fs, line))
        {
            std::stringstream input_line(line);
            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols; j++)
                {
                    input_line >> idata[i][j];
                }
            }
        }
        else
        {
            return false;
        }
        return true;
    }
    return false;
}

/*!
 * \brief Helper function to load the matrix from a file 
 * 
 * \tparam  Tidata data type 
 * \param   idata  array of input data of type Tidata
 * \param   nRows  number of rows
 * \param   nCols  number of columns for each row
 * \param options  (default) 0 load matrix from matrix format and 1 load matrix from vector format
 */
template <typename Tidata>
inline bool loadMatrix(std::fstream &fs, Tidata **idata, size_t nRows, size_t *nCols, size_t options = 0)
{
    std::string line;

    if (options == 0)
    {
        for (int i = 0; i < nRows; i++)
        {
            if (getline(fs, line))
            {
                std::stringstream input_line(line);

                for (int j = 0; j < nCols[i]; j++)
                {
                    input_line >> idata[i][j];
                }
            }
            else
            {
                return false;
            }
        }
        return true;
    }
    else if (options == 1)
    {
        if (getline(fs, line))
        {
            std::stringstream input_line(line);
            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols[i]; j++)
                {
                    input_line >> idata[i][j];
                }
            }
        }
        else
        {
            return false;
        }
        return true;
    }
    return false;
}

template <typename Tidata>
inline bool loadMatrix(std::fstream &fs, Tidata *idata, size_t nRows, size_t nCols)
{
    TEMapX<Tidata> TiMatrix(idata, nRows, nCols);
    return loadMatrix<TEMapX<Tidata>>(fs, TiMatrix);
}

template <typename Tidata>
inline bool loadMatrix(std::fstream &fs, Tidata *idata, size_t nRows)
{
    Eigen::Map<Eigen::Matrix<Tidata, 1, Eigen::Dynamic>> TiMatrix(idata, 1, nRows);
    return loadMatrix<Eigen::Map<Eigen::Matrix<Tidata, 1, Eigen::Dynamic>>>(fs, TiMatrix);
}

/*!
 * \brief Helper function to save the matrix of type TEMX into a file 
 * 
 * \tparam  TEMX   typedef for Eigen matrix 
 * \param   EMX    Eigen matrix
 */
template <typename TEMX>
inline bool saveMatrix(std::fstream &fs, TEMX EMX)
{
    if (!fs.is_open())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "This file stream is not open for writing." << std::endl;
        return false;
    }

    fs << std::fixed;
    fs << EMX.format(fmt);
    fs << fmt.rowSeparator;

    return true;
}

/*!
 * \brief Helper function to save the matrix into a file 
 * 
 * \tparam  Tidata data type 
 * \param   idata  array of input data of type Tidata
 * \param   nRows  number of rows
 * \param   nCols  number of columns
 * \param options  (default) 0 save matrix in matrix format and proceed the position indicator to the next line & 
 *                           1 save matrix in vector format and proceed the position indicator to the next line &
 *                           2 save matrix in vector format and kepp the position indicator on the same line
 */
template <typename Tidata>
inline bool saveMatrix(std::fstream &fs, Tidata **idata, size_t nRows, size_t nCols, size_t options = 0)
{
    if (!fs.is_open())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "This file stream is not open for writing." << std::endl;
        return false;
    }

    std::string rowSeparator;
    if (options > 0)
    {
        rowSeparator = fmt.rowSeparator;
        fmt.rowSeparator = fmt.coeffSeparator;
    }

    if (fs.tellp() == 0)
    {
        if (std::numeric_limits<Tidata>::is_integer)
        {
            fs.precision(0);
        }
        else
        {
            fs.precision(Eigen::NumTraits<Tidata>::digits10());
        }
        fs << std::fixed;

        width = 0;
    }
    else
    {
        width = std::max<Index>(0, width);
    }

    for (Index i = 0; i < nRows; i++)
    {
        for (Index j = 0; j < nCols; j++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << idata[i][j];
            width = std::max<Index>(width, Index(sstr.str().length()));
        }
    }

    if (width)
    {
        for (Index i = 0; i < nRows; ++i)
        {
            fs.width(width);
            fs << idata[i][0];
            for (Index j = 1; j < nCols; ++j)
            {
                fs << fmt.coeffSeparator;
                fs.width(width);
                fs << idata[i][j];
            }
            fs << fmt.rowSeparator;
        }
    }
    else
    {
        for (Index i = 0; i < nRows; ++i)
        {
            fs << idata[i][0];
            for (Index j = 1; j < nCols; ++j)
            {
                fs << fmt.coeffSeparator;
                fs << idata[i][j];
            }
            fs << fmt.rowSeparator;
        }
    }

    if (options == 0)
    {
        return true;
    }
    else if (options == 1)
    {
        fmt.rowSeparator = rowSeparator;
        fs << fmt.rowSeparator;
        return true;
    }
    else if (options == 2)
    {
        fmt.rowSeparator = rowSeparator;
        fs << fmt.coeffSeparator;
        return true;
    }
    return false;
}

/*!
 * \brief Helper function to save the matrix into a file 
 * 
 * \tparam  Tidata data type 
 * \param   idata  array of input data of type Tidata
 * \param   nRows  number of rows
 * \param   *nCols number of columns for each row
 * \param options  (default) 0 saves matrix in matrix format and proceeds the position indicator to the next line & 
 *                           1 saves matrix in vector format and proceeds the position indicator to the next line &
 *                           2 saves matrix in vector format and kepps the position indicator on the same line
 */
template <typename Tidata>
inline bool saveMatrix(std::fstream &fs, Tidata **idata, size_t nRows, size_t *nCols, size_t options = 0)
{
    if (!fs.is_open())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "This file stream is not open for writing." << std::endl;
        return false;
    }

    std::string rowSeparator;
    if (options > 0)
    {
        rowSeparator = fmt.rowSeparator;
        fmt.rowSeparator = fmt.coeffSeparator;
    }

    if (fs.tellp() == 0)
    {
        if (std::numeric_limits<Tidata>::is_integer)
        {
            fs.precision(0);
        }
        else
        {
            fs.precision(Eigen::NumTraits<Tidata>::digits10());
        }
        fs << std::fixed;

        width = 0;
    }
    else
    {
        width = std::max<Index>(0, width);
    }

    for (Index i = 0; i < nRows; i++)
    {
        for (Index j = 0; j < nCols[i]; j++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << idata[i][j];
            width = std::max<Index>(width, Index(sstr.str().length()));
        }
    }

    if (width)
    {
        for (Index i = 0; i < nRows; ++i)
        {
            fs.width(width);
            fs << idata[i][0];
            for (Index j = 1; j < nCols[i]; ++j)
            {
                fs << fmt.coeffSeparator;
                fs.width(width);
                fs << idata[i][j];
            }
            fs << fmt.rowSeparator;
        }
    }
    else
    {
        for (Index i = 0; i < nRows; ++i)
        {
            fs << idata[i][0];
            for (Index j = 1; j < nCols[i]; ++j)
            {
                fs << fmt.coeffSeparator;
                fs << idata[i][j];
            }
            fs << fmt.rowSeparator;
        }
    }

    if (options == 0)
    {
        return true;
    }
    else if (options == 1)
    {
        fmt.rowSeparator = rowSeparator;
        fs << fmt.rowSeparator;
        return true;
    }
    else if (options == 2)
    {
        fmt.rowSeparator = rowSeparator;
        fs << fmt.coeffSeparator;
        return true;
    }
    return false;
}

template <typename Tidata>
inline bool saveMatrix(std::fstream &fs, Tidata *idata, size_t nRows, size_t nCols = 1, size_t options = 0)
{
    if (!fs.is_open())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "This file stream is not open for writing." << std::endl;
        return false;
    }

    std::string rowSeparator;

    if (options > 0)
    {
        rowSeparator = fmt.rowSeparator;
        fmt.rowSeparator = fmt.coeffSeparator;
    }

    if (fs.tellp() == 0)
    {
        if (std::numeric_limits<Tidata>::is_integer)
        {
            fs.precision(0);
        }
        else
        {
            fs.precision(Eigen::NumTraits<Tidata>::digits10());
        }
        fs << std::fixed;

        width = 0;
    }
    else
    {
        width = std::max<Index>(0, width);
    }

    for (Index i = 0; i < nRows * nCols; i++)
    {
        std::stringstream sstr;
        sstr.copyfmt(fs);
        sstr << idata[i];
        width = std::max<Index>(width, Index(sstr.str().length()));
    }

    if (width)
    {
        if (nCols == 1)
        {
            fs.width(width);
            fs << idata[0];
            for (Index i = 1; i < nRows; i++)
            {
                fs << fmt.coeffSeparator;
                fs.width(width);
                fs << idata[i];
            }
            fs << fmt.rowSeparator;
        }
        else
        {
            for (Index i = 0, l = 0; i < nRows; i++)
            {
                fs.width(width);
                fs << idata[l];
                for (Index j = 1; j < nCols; j++)
                {
                    l++;
                    fs << fmt.coeffSeparator;
                    fs.width(width);
                    fs << idata[l];
                }
                l++;
                fs << fmt.rowSeparator;
            }
        }
    }
    else
    {
        if (nCols == 1)
        {
            fs << idata[0];
            for (Index i = 1; i < nRows; i++)
            {
                fs << fmt.coeffSeparator;
                fs << idata[i];
            }
            fs << fmt.rowSeparator;
        }
        else
        {
            for (Index i = 0, l = 0; i < nRows; i++)
            {
                fs << idata[l];
                for (Index j = 1; j < nCols; j++)
                {
                    l++;
                    fs << fmt.coeffSeparator;
                    fs << idata[l];
                }
                l++;
                fs << fmt.rowSeparator;
            }
        }
    }

    if (options == 0)
    {
        return true;
    }
    else if (options == 1)
    {
        fmt.rowSeparator = rowSeparator;
        fs << fmt.rowSeparator;
        return true;
    }
    else if (options == 2)
    {
        fmt.rowSeparator = rowSeparator;
        return true;
    }
    return false;
}
#endif
