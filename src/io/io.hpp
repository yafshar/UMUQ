#ifndef UMUQ_IO_H
#define UMUQ_IO_H

#include "../core/digits10.hpp"

/*! \class ioFormat
 * \ingroup io
 * 
 * \brief Stores a set of parameters controlling the way matrices are printed
 *
 * List of available parameters:
 *  - \b coeffSeparator string printed between two coefficients of the same row
 *  - \b rowSeparator   string printed between two rows
 *  - \b rowPrefix      string printed at the beginning of each row
 *  - \b rowSuffix      string printed at the end of each row
 *
 */
struct ioFormat
{
    /*!
     * Default constructor, see ioFormat for the meaning of the parameters 
     */
    ioFormat(const std::string &_coeffSeparator = " ",
             const std::string &_rowSeparator = "\n",
             const std::string &_rowPrefix = "",
             const std::string &_rowSuffix = "") : coeffSeparator(_coeffSeparator),
                                                   rowSeparator(_rowSeparator),
                                                   rowPrefix(_rowPrefix),
                                                   rowSuffix(_rowSuffix) {}

    std::string coeffSeparator;
    std::string rowSeparator;
    std::string rowPrefix;
    std::string rowSuffix;
};

/*! \class io
 * \ingroup io
 * 
 * \brief This class includes IO functionality.
 *
 * List of available functions:
 * - \b isFileOpened  Check if the file is opened
 * - \b isFileExist   Check if the file exists
 * 
 * - \b openFile      Opens the file with the vailable file open flags:
 *      - \b app      Seeks to the end of stream before each write
 *      - \b binary   Open in binary mode
 *      - \b in       Open for reading
 *      - \b out      Open for writing
 *      - \b trunc    Discard the contents of the stream when opening
 *      - \b ate	  Seeks to the end of stream immediately after open 
 *  
 * - \b readLine      Get the string from stream file
 * - \b rewindFile    Set the position of stream to the beginning of a file
 * - \b closeFile     Close the file
 * - \b getFstream    Get the stream
 * - \b getLine       Get the Line 
 * - \b saveMatrix    Helper function to save the matrix into a file
 * - \b loadMatrix    Helper function to load a matrix from a file
 * - \b printMatrix   Helper function to print the matrix to the output stream
 * 
 */
class io
{
  public:
    static const std::ios_base::openmode app = std::fstream::app;
    static const std::ios_base::openmode binary = std::fstream::binary;
    static const std::ios_base::openmode in = std::fstream::in;
    static const std::ios_base::openmode out = std::fstream::out;
    static const std::ios_base::openmode ate = std::fstream::ate;
    static const std::ios_base::openmode trunc = std::fstream::trunc;

    /*!
     * \brief Construct a new io object
     * 
     */
    io() {}

    /*!
     * \brief Destroy the io object
     * 
     */
    ~io()
    {
        closeFile();
    }

    /*!
     * \brief   It is true if the file is opened
     * 
     * \returns true if the file is already opened 
     */
    inline bool isFileOpened() const { return fs.is_open(); }

    /*!
     * \brief Check to see whether the file fileName exists and accessible to read or write!
     *  
     * \returns true if the file exists 
     */
    inline bool isFileExist(const char *fileName)
    {
        struct stat buffer;
        return (stat(fileName, &buffer) == 0);
    }

    /*!
     * \brief Opens the file whose name is specified with the parameter filename 
     *  
     * Opens the file whose name is specified in the parameter filename and
     * associates it with a stream that can be identified in future operations 
     * by the FILE pointer returned.inline   
     * 
     * Available file open flags:
     * - \b std::fstream::app 	  Seeks to the end of stream before each write
     * - \b std::fstream::binary  Open in binary mode
     * - \b std::fstream::in 	  Open for reading
     * - \b std::fstream::out 	  Open for writing
     * - \b std::fstream::trunc   Discard the contents of the stream when opening
     * - \b std::fstream::ate 	  Seeks to the end of stream immediately after open
     * 
     * \returns true if everything goes OK
     */
    bool openFile(const char *fileName, const std::ios_base::openmode mode = in)
    {
        if (fs.is_open())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Pointer to the File '" << fileName << "' is busy!" << std::endl;
            return false;
        }

        fs.open(fileName, mode);
        if (!fs.is_open())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "'" << fileName << "' does not exists!" << std::endl;
            return false;
        }

        //! Returns true if an error has occurred on the associated stream.
        if (fs.fail())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "An error has occurred on the associated stream from opening '" << fileName << "' ." << std::endl;
            return false;
        }

        return true;
    }

    /*!
     * \brief Get string from stream
     * 
     * Get a string from stream and stores them into line until 
     * a newline or the end-of-file is reached, whichever happens first.
     * 
     * \returns true if no error occurs on the associated stream
     */
    bool readLine(const char comment = '#')
    {
        std::string linetmp;
        for (;;)
        {
            std::getline(fs, linetmp);
            if (fs.good())
            {
                const std::string::size_type linePos = linetmp.find_first_not_of(" \t\n");

                // See if we found a valid line
                if (linetmp.length() > 0 && linetmp[linePos] != comment)
                {
                    //Trim the empty space at the start of the line
                    line = linetmp.substr(linePos);
                    return true;
                }
            }
            else
            {
                linetmp.clear();
                return false;
            }
        }
    }

    /*!
     * \brief Set position of stream to the beginning
     * Sets the position indicator associated with stream to the beginning of the file.
     */
    inline void rewindFile()
    {
        //clearing all error state flags if there is any
        fs.clear();

        //!Rewind the file
        fs.seekg(0);
        return;
    }

    /*!
     * \brief Close the File
     * 
     */
    inline void closeFile()
    {
        fs.close();
        return;
    }

    /*!
     * \brief Get the stream
     * 
     */
    inline std::fstream &getFstream() { return fs; }

    /*!
     * \brief Get the Line object
     * 
     * \return std::string& 
     */
    inline std::string &getLine() { return line; }

    /*!
     * \brief Helper function to save the matrix of type TM with TF format into a file 
     * 
     * \tparam  TM    Matrix type 
     * \tparam  TF    IO format type (We can use either of ioFormat or Eigen::IOFormat)
     * 
     * \param   MX    Input matrix of data
     * \param   IOfmt IO format for the matrix type
     *
     * \returns true if no error occurs during writing the matrix
     */
    template <typename TM, typename TF>
    bool saveMatrix(TM MX, TF const IOfmt)
    {
        if (!fs.is_open())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "This file stream is not open for writing." << std::endl;
            return false;
        }

        fs << std::fixed;
        fs << MX.format(IOfmt);
        fs << fmt.rowSeparator;

        return true;
    }

    /*!
     * \brief Helper function to save the matrix into a file 
     * 
     * \tparam  TD     Data type 
     * 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns
     * \param options  (Default) 0 Save matrix in matrix format and proceed the position indicator to the next line & 
     *                           1 Save matrix in vector format and proceed the position indicator to the next line &
     *                           2 Save matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true if no error occurs during writing the matrix
     */
    template <typename TD>
    bool saveMatrix(TD **idata, const int nRows, const int nCols, const int options = 0)
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

        //!IF the output position indicator of the current associated streambuf object is at the startline
        if (fs.tellp() == 0)
        {
            if (std::numeric_limits<TD>::is_integer)
            {
                //!Manages the precision (i.e. how many digits are generated)
                fs.precision(0);
            }
            else
            {
                //!Manages the precision (i.e. how many digits are generated)
                fs.precision(digits10<TD>());
            }
            fs << std::fixed;

            Width = 0;
        }
        else
        {
            Width = std::max<std::ptrdiff_t>(0, Width);
        }

        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(fs);
                sstr << idata[i][j];
                Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
            }
        }

        if (Width)
        {
            for (int i = 0; i < nRows; ++i)
            {
                fs.width(Width);
                fs << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << idata[i][j];
                }
                fs << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0; i < nRows; ++i)
            {
                fs << idata[i][0];
                for (int j = 1; j < nCols; ++j)
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
     * \tparam  TD     Data type 
     * 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns for each row
     * \param options  (Default) 0 Saves matrix in matrix format and proceeds the position indicator to the next line & 
     *                           1 Saves matrix in vector format and proceeds the position indicator to the next line &
     *                           2 Saves matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true if no error occurs during writing the matrix
     */
    template <typename TD>
    bool saveMatrix(TD **idata, const int nRows, const int *nCols, const int options = 0)
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

        //!IF the output position indicator of the current associated streambuf object is at the startline
        if (fs.tellp() == 0)
        {
            if (std::numeric_limits<TD>::is_integer)
            {
                //!Manages the precision (i.e. how many digits are generated)
                fs.precision(0);
            }
            else
            {
                //!Manages the precision (i.e. how many digits are generated)
                fs.precision(digits10<TD>());
            }
            fs << std::fixed;

            Width = 0;
        }
        else
        {
            Width = std::max<std::ptrdiff_t>(0, Width);
        }

        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols[i]; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(fs);
                sstr << idata[i][j];
                Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
            }
        }

        if (Width)
        {
            for (int i = 0; i < nRows; ++i)
            {
                fs.width(Width);
                fs << idata[i][0];
                for (int j = 1; j < nCols[i]; ++j)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << idata[i][j];
                }
                fs << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0; i < nRows; ++i)
            {
                fs << idata[i][0];
                for (int j = 1; j < nCols[i]; ++j)
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
     * \tparam TD      Data type 
     * 
     * \param idata    Array of input data of type TD
     * \param nRows    Number of rows 
     * \param nCols    Number of columns for each row (default is 1)
     * \param options  (Default) 0 Saves matrix in matrix format and proceeds the position indicator to the next line & 
     *                           1 Saves matrix in vector format and proceeds the position indicator to the next line &
     *                           2 Saves matrix in vector format and keep the position indicator on the same line
     */
    template <typename TD>
    bool saveMatrix(TD *idata, const int nRows, const int nCols = 1, const int options = 0)
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
        //!IF the output position indicator of the current associated streambuf object is at the startline
        if (fs.tellp() == 0)
        {
            if (std::numeric_limits<TD>::is_integer)
            {
                //!Manages the precision (i.e. how many digits are generated)
                fs.precision(0);
            }
            else
            {
                //!Manages the precision (i.e. how many digits are generated)
                fs.precision(digits10<TD>());
            }
            fs << std::fixed;

            Width = 0;
        }
        else
        {
            Width = std::max<std::ptrdiff_t>(0, Width);
        }

        for (int i = 0; i < nRows * nCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
        }

        if (Width)
        {
            if (nCols == 1)
            {
                fs.width(Width);
                fs << idata[0];
                for (int i = 1; i < nRows; i++)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << idata[i];
                }
                fs << fmt.rowSeparator;
            }
            else
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    fs.width(Width);
                    fs << idata[l];
                    for (int j = 1; j < nCols; j++)
                    {
                        l++;
                        fs << fmt.coeffSeparator;
                        fs.width(Width);
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
                for (int i = 1; i < nRows; i++)
                {
                    fs << fmt.coeffSeparator;
                    fs << idata[i];
                }
                fs << fmt.rowSeparator;
            }
            else
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    fs << idata[l];
                    for (int j = 1; j < nCols; j++)
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

    /*!
     * \brief Helper function to save the matrix into a file 
     * 
     * \tparam TD          Data type 
     * 
     * \param idata        Array of input data of type TD
     * \param idata        Array of input data of type TD
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type TD
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows
     */
    template <typename TD>
    bool saveMatrix(TD *idata, const int idataCols, TD *ifvalue, const int ifvalueCols, const int nRows)
    {
        if (!fs.is_open())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "This file stream is not open for writing." << std::endl;
            return false;
        }

        //!IF the output position indicator of the current associated streambuf object is at the startline
        if (fs.tellp() == 0)
        {
            if (std::numeric_limits<TD>::is_integer)
            {
                //!Manages the precision (i.e. how many digits are generated)
                fs.precision(0);
            }
            else
            {
                //!Manages the precision (i.e. how many digits are generated)
                fs.precision(digits10<TD>());
            }
            fs << std::fixed;

            Width = 0;
        }
        else
        {
            Width = std::max<std::ptrdiff_t>(0, Width);
        }

        for (int i = 0; i < nRows * idataCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
        }

        for (int i = 0; i < nRows * ifvalueCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << ifvalue[i];
            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
        }

        if (Width)
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                fs.width(Width);
                fs << idata[l++];
                for (int j = 1; j < idataCols; j++)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << idata[l++];
                }
                for (int j = 0; j < ifvalueCols; j++)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << ifvalue[k++];
                }
                fs << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                fs << idata[l++];
                for (int j = 1; j < idataCols; j++)
                {
                    fs << fmt.coeffSeparator;
                    fs << idata[l++];
                }
                for (int j = 0; j < ifvalueCols; j++)
                {
                    fs << fmt.coeffSeparator;
                    fs << ifvalue[k++];
                }
                fs << fmt.rowSeparator;
            }
        }

        return true;
    }

    /*!
     * \brief Helper function to load the matrix of type TM from a file 
     * 
     * \tparam  TM   Matrix type
     * 
     * \param   MX   Input matrix of data
     *
     * \returns true if no error occurs during reading a matrix
     */
    template <typename TM>
    bool loadMatrix(TM &MX)
    {
        std::string Line;

        for (int i = 0; i < MX.rows(); i++)
        {
            if (std::getline(fs, Line))
            {
                std::stringstream inLine(Line);

                for (int j = 0; j < MX.cols(); j++)
                {
                    inLine >> MX(i, j);
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
     * \tparam  TD     Data type 
     * 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns
     * \param options  (default) 0 load matrix from matrix format and 1 load matrix from vector format
     *
     * \returns true if no error occurs during reading data
     */
    template <typename TD>
    bool loadMatrix(TD **idata, const int nRows, const int nCols, const int options = 0)
    {
        std::string Line;

        if (options == 0)
        {
            for (int i = 0; i < nRows; i++)
            {
                if (std::getline(fs, Line))
                {
                    std::stringstream inLine(Line);

                    for (int j = 0; j < nCols; j++)
                    {
                        inLine >> idata[i][j];
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
            if (std::getline(fs, Line))
            {
                std::stringstream inLine(Line);
                for (int i = 0; i < nRows; i++)
                {
                    for (int j = 0; j < nCols; j++)
                    {
                        inLine >> idata[i][j];
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
     * \tparam  TD     Data type 
     * 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns for each row
     * \param options  (Default) 0 load matrix from matrix format and 1 load matrix from vector format
     *
     * \returns true if no error occurs during reading data
     */
    template <typename TD>
    bool loadMatrix(TD **idata, const int nRows, const int *nCols, const int options = 0)
    {
        std::string Line;

        if (options == 0)
        {
            for (int i = 0; i < nRows; i++)
            {
                if (std::getline(fs, Line))
                {
                    std::stringstream inLine(Line);

                    for (int j = 0; j < nCols[i]; j++)
                    {
                        inLine >> idata[i][j];
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
            if (std::getline(fs, Line))
            {
                std::stringstream inLine(Line);
                for (int i = 0; i < nRows; i++)
                {
                    for (int j = 0; j < nCols[i]; j++)
                    {
                        inLine >> idata[i][j];
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
     * \tparam  TD     Data type 
     * 
     * \param   idata  Input array of data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns for each row (default is 1)
     *
     * \returns true if no error occurs during reading data
     */
    template <typename TD>
    bool loadMatrix(TD *idata, const int nRows, const int nCols = 1)
    {
        std::string Line;

        if (nCols == 1)
        {
            if (std::getline(fs, Line))
            {
                std::stringstream inLine(Line);
                for (int i = 0; i < nRows; i++)
                {
                    inLine >> idata[i];
                }
            }
            else
            {
                return false;
            }
            return true;
        }
        else
        {
            for (int i = 0, l = 0; i < nRows; i++)
            {
                if (std::getline(fs, Line))
                {
                    std::stringstream inLine(Line);

                    for (int j = 0; j < nCols; j++, l++)
                    {
                        inLine >> idata[l];
                    }
                }
                else
                {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam TD          Data type 
     * 
     * \param idata        Input array of data of type TD
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type TD
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows
     * 
     * \returns true if no error occurs during reading data
     */
    template <typename TD>
    bool loadMatrix(TD *idata, const int idataCols, TD *ifvalue, const int ifvalueCols, const int nRows)
    {
        std::string Line;

        for (int i = 0, k = 0, l = 0; i < nRows; i++)
        {
            if (std::getline(fs, Line))
            {
                std::stringstream inLine(Line);

                for (int j = 0; j < idataCols; j++, k++)
                {
                    inLine >> idata[k];
                }

                for (int j = 0; j < ifvalueCols; j++, l++)
                {
                    inLine >> ifvalue[k];
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
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD     Data type
     * 
     * \param   title  Title (string) that should be written at the top 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns
     */
    template <typename TD>
    void printMatrix(const char *title, TD **idata, const int nRows, const int nCols)
    {
        std::string sep = "\n----------------------------------------\n";
        std::cout << sep;
        if (std::strlen(title) > 0)
        {
            std::cout << title << "\n\n";
        }

        if (std::numeric_limits<TD>::is_integer)
        {
            //!Manages the precision (i.e. how many digits are generated)
            std::cout.precision(0);
        }
        else
        {
            //!Manages the precision (i.e. how many digits are generated)
            std::cout.precision(digits10<TD>());
        }
        std::cout << std::fixed;

        Width = 0;

        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(std::cout);
                sstr << idata[i][j];
                Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
            }
        }

        if (Width)
        {
            for (int i = 0; i < nRows; ++i)
            {
                std::cout.width(Width);
                std::cout << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << idata[i][j];
                }
                std::cout << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0; i < nRows; ++i)
            {
                std::cout << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout << idata[i][j];
                }
                std::cout << fmt.rowSeparator;
            }
        }
        std::cout << sep;
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam TD    TD type of data
     * 
     * \param idata  Array of input data of type TD
     * \param nRows  Number of rows
     * \param nCols  Number of columns
     */
    template <typename TD>
    void printMatrix(TD **idata, const int nRows, const int nCols)
    {
        printMatrix<TD>("", idata, nRows, nCols);
    }
    
    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD     Data type
     * 
     * \param   title  Title (string) that should be written at the top 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns (default is 1)
     */
    template <typename TD>
    void printMatrix(const char *title, TD *idata, const int nRows, const int nCols = 1)
    {
        std::string sep = "\n----------------------------------------\n";
        std::cout << sep;
        if (std::strlen(title) > 0)
        {
            std::cout << title << "\n\n";
        }

        if (std::numeric_limits<TD>::is_integer)
        {
            //!Manages the precision (i.e. how many digits are generated)
            std::cout.precision(0);
        }
        else
        {
            //!Manages the precision (i.e. how many digits are generated)
            std::cout.precision(digits10<TD>());
        }
        std::cout << std::fixed;

        Width = 0;

        for (int i = 0; i < nRows * nCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
        }

        if (nCols == 1)
        {
            if (Width)
            {
                std::cout.width(Width);
                std::cout << idata[0];
                for (int i = 1; i < nRows; i++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << idata[i];
                }
            }
            else
            {
                std::cout << idata[0];
                for (int i = 1; i < nRows; i++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout << idata[i];
                }
            }
            std::cout << fmt.rowSeparator;
            std::cout << sep;
        }
        else
        {
            if (Width)
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    std::cout.width(Width);
                    std::cout << idata[l];
                    for (int j = 1; j < nCols; j++, l++)
                    {
                        std::cout << fmt.coeffSeparator;
                        std::cout.width(Width);
                        std::cout << idata[l];
                    }
                    std::cout << fmt.rowSeparator;
                }
            }
            else
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    std::cout << idata[l];
                    for (int j = 1; j < nCols; j++, l++)
                    {
                        std::cout << fmt.coeffSeparator;
                        std::cout << idata[l];
                    }
                    std::cout << fmt.rowSeparator;
                }
            }
            std::cout << sep;
        }
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD     Data type
     * 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns (default is 1)
     */
    template <typename TD>
    void printMatrix(TD *idata, const int nRows, const int nCols = 1)
    {
        printMatrix<TD>("", idata, nRows, nCols);
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam TD            Data type
     * 
     * \param title          Title (string) that should be written at the top 
     * \param idata          Array of input data of type TD
     * \param idataCols      Number of columns of inpput array data (idata)
     * \param ifvalue        Array of input value data of type TD
     * \param ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param nRows          Number of rows
     */
    template <typename TD>
    void printMatrix(const char *title, TD *idata, const int idataCols, TD *ifvalue, const int ifvalueCols, const int nRows)
    {
        std::string sep = "\n----------------------------------------\n";
        std::cout << sep;
        if (std::strlen(title) > 0)
        {
            std::cout << title << "\n\n";
        }

        if (std::numeric_limits<TD>::is_integer)
        {
            //!Manages the precision (i.e. how many digits are generated)
            std::cout.precision(0);
        }
        else
        {
            //!Manages the precision (i.e. how many digits are generated)
            std::cout.precision(digits10<TD>());
        }
        std::cout << std::fixed;

        Width = 0;

        for (int i = 0; i < nRows * idataCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
        }

        for (int i = 0; i < nRows * ifvalueCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << ifvalue[i];
            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
        }

        if (Width)
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                std::cout.width(Width);
                std::cout << idata[l++];
                for (int j = 1; j < idataCols; j++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << idata[l++];
                }
                for (int j = 0; j < ifvalueCols; j++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << ifvalue[k++];
                }
                std::cout << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                std::cout << idata[l++];
                for (int j = 1; j < idataCols; j++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout << idata[l++];
                }
                for (int j = 0; j < ifvalueCols; j++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout << ifvalue[k++];
                }
                std::cout << fmt.rowSeparator;
            }
        }
    }
    
    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam TD            Data type
     * 
     * \param idata          Array of input data of type TD
     * \param idataCols      Number of columns of inpput array data (idata)
     * \param ifvalue        Array of input value data of type TD
     * \param ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param nRows          Number of rows
     */
    template <typename TD>
    void printMatrix(TD *idata, const int idataCols, TD *ifvalue, const int ifvalueCols, const int nRows)
    {
        printMatrix<TD>("", idata, idataCols, ifvalue, ifvalueCols, nRows);
    }

  private:
    //! Input/output operations on file based streams
    std::fstream fs;

    //! Line for reading the string of data
    std::string line;

    //! Index
    typedef std::ptrdiff_t Idx;

    //! Width parameter of the stream out or in
    std::ptrdiff_t Width;

    //! IO format
    ioFormat fmt;
};

#endif
