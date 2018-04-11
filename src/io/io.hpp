#ifndef UMHBM_IO_H
#define UMHBM_IO_H

#include "../core/digits10.hpp"

/*! 
  * \brief Stores a set of parameters controlling the way matrices are printed
  *
  * List of available parameters:
  *  - \b coeffSeparator string printed between two coefficients of the same row
  *  - \b rowSeparator string printed between two rows
  *  - \b rowPrefix string printed at the beginning of each row
  *  - \b rowSuffix string printed at the end of each row
  *
  */
struct IOFormat
{
    /** Default constructor, see IOFormat for the meaning of the parameters */
    IOFormat(const std::string &_coeffSeparator = " ",
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
*   \brief io is a class which includes some IO functionality.
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

    //!default constrcutor
    io(){};

    ~io()
    {
        closeFile();
    };

    /*!
     * \brief return true if file is opened
     */
    inline bool isFileOpened() const { return fs.is_open(); }

    /*!
     * \brief Check to see whether the file fileName exists and accessible to read or write!
     *  
     */
    inline bool isFileExist(const char *fileName)
    {
        struct stat buffer;
        return (stat(fileName, &buffer) == 0);
    }

    /*!
     * \brief Opens the file whose name is specified in the parameter filename 
     *  
     * Opens the file whose name is specified in the parameter filename and
     * associates it with a stream that can be identified in future operations 
     * by the FILE pointer returned.inline   
     * 
     * stream open mode type
     * 
     * std::fstream::app 	  seek to the end of stream before each write
     * std::fstream::binary   open in binary mode
     * std::fstream::in 	  open for reading
     * std::fstream::out 	  open for writing
     * std::fstream::trunc 	  discard the contents of the stream when opening
     * std::fstream::ate 	  seek to the end of stream immediately after open
     */
    inline bool openFile(const char *fileName, const std::ios_base::openmode mode = in)
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
     * Get string from stream and stores them into line until (LINESIZE-1) characters 
     * have been read or either a newline or the end-of-file is reached, whichever happens first.
     */
    inline bool readLine(const char comment = '#')
    {
        std::string linetmp;
        for (;;)
        {
            std::getline(fs, linetmp);
            if (fs.good())
            {
                std::string::size_type linePos = linetmp.find_first_not_of(" \t\n");

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
     */
    inline void closeFile()
    {
        fs.close();
        return;
    }

    /*!
     * \brief Get the stream
     */
    std::fstream &getFstream()
    {
        return fs;
    }

    std::string &getLine()
    {
        return line;
    }

    /*!
     * \brief Helper function to save the matrix of type TM with TF format into a file 
     * 
     * \tparam  TM    typedef for matrix 
     * \tparam  TF    typedef for format of writing
     * \param   MX    matrix
     * \param   IOfmt IO format for the matrix type
     */
    template <typename TM, typename TF>
    inline bool saveMatrix(TM MX, TF IOfmt)
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
     * \tparam  TD     data type 
     * \param   idata  array of input data of type TD
     * \param   nRows  number of rows
     * \param   nCols  number of columns
     * \param options  (default) 0 save matrix in matrix format and proceed the position indicator to the next line & 
     *                           1 save matrix in vector format and proceed the position indicator to the next line &
     *                           2 save matrix in vector format and kepp the position indicator on the same line
     */
    template <typename TD>
    inline bool saveMatrix(TD **idata, const int nRows, const int nCols, const int options = 0)
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
     * \tparam  TD     data type 
     * \param   idata  array of input data of type TD
     * \param   nRows  number of rows
     * \param   *nCols number of columns for each row
     * \param options  (default) 0 saves matrix in matrix format and proceeds the position indicator to the next line & 
     *                           1 saves matrix in vector format and proceeds the position indicator to the next line &
     *                           2 saves matrix in vector format and kepps the position indicator on the same line
     */
    template <typename TD>
    inline bool saveMatrix(TD **idata, const int nRows, const int *nCols, const int options = 0)
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

    template <typename TD>
    inline bool saveMatrix(TD *idata, const int nRows, const int nCols = 1, const int options = 0)
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
     * \brief Helper function to load the matrix of type TM from a file 
     * 
     * \tparam  TM   typedef for matrix 
     * \param   MX   Matrix
     */
    template <typename TM>
    inline bool loadMatrix(TM &MX)
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
     * \tparam  TD data type 
     * \param   idata  array of input data of type TD
     * \param   nRows  number of rows
     * \param   nCols  number of columns
     * \param options  (default) 0 load matrix from matrix format and 1 load matrix from vector format
     */
    template <typename TD>
    inline bool loadMatrix(TD **idata, const int nRows, const int nCols, const int options = 0)
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
     * \tparam  TD data type 
     * \param   idata  array of input data of type TD
     * \param   nRows  number of rows
     * \param   nCols  number of columns for each row
     * \param options  (default) 0 load matrix from matrix format and 1 load matrix from vector format
     */
    template <typename TD>
    inline bool loadMatrix(TD **idata, const int nRows, const int *nCols, const int options = 0)
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

    template <typename TD>
    inline bool loadMatrix(TD *idata, const int nRows, const int nCols = 1)
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
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD type of data
     * 
     * \param   title  string that should be written at the top 
     * \param   idata  array of input data of type TD
     * \param   nRows  number of rows
     * \param   nCols  number of columns
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

    template <typename TD>
    void printMatrix(TD **idata, const int nRows, const int nCols)
    {
        printMatrix<TD>("", idata, nRows, nCols);
    }

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

    template <typename TD>
    void printMatrix(TD *idata, const int nRows, const int nCols = 1)
    {
        printMatrix<TD>("", idata, nRows, nCols);
    }

  private:
    //Input/output operations on file based streams
    std::fstream fs;
    //Line for reading the string of data
    std::string line;

    typedef std::ptrdiff_t Idx;
    std::ptrdiff_t Width;

    //IO format
    IOFormat fmt;
};

#endif
