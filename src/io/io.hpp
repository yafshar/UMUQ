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
    ioFormat(std::string const &_coeffSeparator = " ",
             std::string const &_rowSeparator = "\n",
             std::string const &_rowPrefix = "",
             std::string const &_rowSuffix = "") : coeffSeparator(_coeffSeparator),
                                                   rowSeparator(_rowSeparator),
                                                   rowPrefix(_rowPrefix),
                                                   rowSuffix(_rowSuffix) {}

    std::string coeffSeparator;
    std::string rowSeparator;
    std::string rowPrefix;
    std::string rowSuffix;

    /*!
     * \brief Operator ==
     * 
     * \param rhs  
     * \return true 
     * \return false 
     */
    inline bool operator==(ioFormat const &rhs)
    {
        return coeffSeparator == rhs.coeffSeparator &&
               rowSeparator == rhs.rowSeparator &&
               rowPrefix == rhs.rowPrefix &&
               rowSuffix == rhs.rowSuffix;
    }

    inline bool operator==(ioFormat const &rhs) const
    {
        return coeffSeparator == rhs.coeffSeparator &&
               rowSeparator == rhs.rowSeparator &&
               rowPrefix == rhs.rowPrefix &&
               rowSuffix == rhs.rowSuffix;
    }

    /*!
     * \brief Operator !=
     * 
     * \param rhs 
     * \return true 
     * \return false 
     */
    inline bool operator!=(ioFormat const &rhs)
    {
        return coeffSeparator != rhs.coeffSeparator ||
               rowSeparator != rhs.rowSeparator ||
               rowPrefix != rhs.rowPrefix ||
               rowSuffix != rhs.rowSuffix;
    }

    inline bool operator!=(ioFormat const &rhs) const
    {
        return coeffSeparator != rhs.coeffSeparator ||
               rowSeparator != rhs.rowSeparator ||
               rowPrefix != rhs.rowPrefix ||
               rowSuffix != rhs.rowSuffix;
    }
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
 * - \b setPrecision  Set the stream Precision
 * - \b setWidth      Set the width parameter of the stream to exactly n.
 * - \b getWidth      Get the width parameter of the input data and stream for the precision 
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
    io() : Width(0), FixedWidth(false) {}

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

        //! Returns false if an error has occurred on the associated stream.
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
     * \brief Set the stream Precision 
     * 
     * \tparam TD Data type
     * 
     * \param  os File based streams
     */
    template <typename TD>
    inline void setPrecision(std::ostream &os)
    {
        if (std::numeric_limits<TD>::is_integer)
        {
            //!Manages the precision (i.e. how many digits are generated)
            os.precision(0);
        }
        else
        {
            //!Manages the precision (i.e. how many digits are generated)
            os.precision(digits10<TD>());
            os << std::fixed;
        }
    }

    /*!
     * \brief Set the width parameter of the stream to exactly n.
     * 
     * If Input Width_ is < 0 the function will set the stream to zero and its setting flag to false 
     * 
     * \param Width_ New value for Width 
     */
    inline void setWidth(int Width_ = 0)
    {
        Width = Width_ < 0 ? std::ptrdiff_t{} : static_cast<std::ptrdiff_t>(Width_);
        FixedWidth = Width_ >= 0;
    }

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam TD  data type
     * 
     * \param idata  Input array of data
     * \param nRows  Number of Rows
     * \param nCols  Number of Columns
     * \param  os    File based streams
     * 
     * \returns the width
     */
    template <typename TD>
    int getWidth(TD *idata, int const nRows, int const nCols, std::ostream &os)
    {
        std::ptrdiff_t tWidth(0);
        setPrecision<TD>(os);
        for (int i = 0; i < nRows * nCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(os);
            sstr << idata[i];
            tWidth = std::max<std::ptrdiff_t>(tWidth, Idx(sstr.str().length()));
        }
        return static_cast<int>(tWidth);
    }

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam TD  data type
     * 
     * \param idata  Input array of data
     * \param nRows  Number of Rows
     * \param nCols  Number of Columns
     * \param  os    File based streams
     * 
     * \returns the width
     */
    template <typename TD>
    int getWidth(std::unique_ptr<TD[]> const &idata, int const nRows, int const nCols, std::ostream &os)
    {
        std::ptrdiff_t tWidth(0);
        setPrecision<TD>(os);
        for (int i = 0; i < nRows * nCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(os);
            sstr << idata[i];
            tWidth = std::max<std::ptrdiff_t>(tWidth, Idx(sstr.str().length()));
        }
        return static_cast<int>(tWidth);
    }

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam TD  data type
     * 
     * \param idata  Input array of data
     * \param nRows  Number of Rows
     * \param nCols  Number of Columns
     * \param  os    File based streams
     * 
     * \returns the width
     */
    template <typename TD>
    int getWidth(TD **idata, int const nRows, int const nCols, std::ostream &os)
    {
        std::ptrdiff_t tWidth(0);
        setPrecision<TD>(os);
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(os);
                sstr << idata[i][j];
                tWidth = std::max<std::ptrdiff_t>(tWidth, Idx(sstr.str().length()));
            }
        }
        return static_cast<int>(tWidth);
    }

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
    bool saveMatrix(TM &MX, TF const &IOfmt)
    {
        if (fs.is_open())
        {
            fs << std::fixed;
            fs << MX.format(IOfmt);
            fs << fmt.rowSeparator;

            return true;
        }

        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "This file stream is not open for writing." << std::endl;
        return false;
    }

    /*!
     * \brief Helper function to save one matrix into a file 
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
    bool saveMatrix(TD **idata, int const nRows, int const nCols, int const options = 0)
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

        setPrecision<TD>(fs);

        if (!FixedWidth)
        {
            Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

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
        }

        if (Width)
        {
            for (int i = 0; i < nRows; ++i)
            {
                fs << fmt.rowPrefix;
                fs.width(Width);
                fs << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << idata[i][j];
                }
                fs << fmt.rowSuffix;
                fs << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0; i < nRows; ++i)
            {
                fs << fmt.rowPrefix;
                fs << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    fs << fmt.coeffSeparator;
                    fs << idata[i][j];
                }
                fs << fmt.rowSuffix;
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
     * \brief Helper function to save one matrix (or entries number matrices) into a file 
     * 
     * \tparam  TD     Data type 
     * 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns for each row
     * \param options  (Default) 0 Saves matrix in matrix format and proceeds the position indicator to the next line & 
     *                           1 Saves matrix in vector format and proceeds the position indicator to the next line &
     *                           2 Saves matrix in vector format and keep the position indicator on the same line
     * \param  entries           Number of data entry (Input data contains pointer to array of data)
     * \param  form              Print format for each row 
     * 
     * \returns true if no error occurs during writing the matrix
     */
    template <typename TD>
    bool saveMatrix(TD **idata, int const nRows, int const *nCols, int const options = 0, int const entries = 1, std::vector<ioFormat> const &form = std::vector<ioFormat>())
    {
        if (!fs.is_open())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "This file stream is not open for writing." << std::endl;
            return false;
        }

        setPrecision<TD>(fs);

        //Default case, only one set of data
        if (entries == 1)
        {
            if (form.size() != nRows)
            {
                std::string rowSeparator;
                if (options > 0)
                {
                    rowSeparator = fmt.rowSeparator;
                    fmt.rowSeparator = fmt.coeffSeparator;
                }

                if (!FixedWidth)
                {
                    Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

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
                }

                if (Width)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        fs << fmt.rowPrefix;
                        fs.width(Width);
                        fs << idata[i][0];
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            fs << fmt.coeffSeparator;
                            fs.width(Width);
                            fs << idata[i][j];
                        }
                        fs << fmt.rowSuffix;
                        fs << fmt.rowSeparator;
                    }
                }
                else
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        fs << fmt.rowPrefix;
                        fs << idata[i][0];
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            fs << fmt.coeffSeparator;
                            fs << idata[i][j];
                        }
                        fs << fmt.rowSuffix;
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
            else
            {
                if (!FixedWidth)
                {
                    Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

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
                }

                if (Width)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        fs << form[i].rowPrefix;
                        fs.width(Width);
                        fs << idata[i][0];
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            fs << form[i].coeffSeparator;
                            fs.width(Width);
                            fs << idata[i][j];
                        }
                        fs << form[i].rowSuffix;
                        fs << form[i].rowSeparator;
                    }
                }
                else
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        fs << form[i].rowPrefix;
                        fs << idata[i][0];
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            fs << form[i].coeffSeparator;
                            fs << idata[i][j];
                        }
                        fs << form[i].rowSuffix;
                        fs << form[i].rowSeparator;
                    }
                }

                return true;
            }
        }
        else
        {
            if (form.size() != nRows)
            {
                std::string rowSeparator;
                if (options > 0)
                {
                    rowSeparator = fmt.rowSeparator;
                    fmt.rowSeparator = fmt.coeffSeparator;
                }

                if (!FixedWidth)
                {
                    Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

                    for (int i = 0; i < nRows; i++)
                    {
                        TD *ePointer = idata[i];
                        for (int e = 0; e < entries; e++)
                        {
                            for (int j = 0; j < nCols[i]; j++)
                            {
                                std::stringstream sstr;
                                sstr.copyfmt(fs);
                                sstr << *ePointer++;
                                Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
                            }
                        }
                    }
                }

                if (Width)
                {
                    TD *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = idata[i] + e * nCols[i];

                            fs << fmt.rowPrefix;
                            fs.width(Width);
                            fs << *ePointer++;
                            for (int j = 1; j < nCols[i]; ++j)
                            {
                                fs << fmt.coeffSeparator;
                                fs.width(Width);
                                fs << *ePointer++;
                            }
                            fs << fmt.rowSuffix;
                            fs << fmt.rowSeparator;
                        }
                        fs << rowSeparator;
                    }
                }
                else
                {
                    TD *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = idata[i] + e * nCols[i];

                            fs << fmt.rowPrefix;
                            fs << *ePointer++;
                            for (int j = 1; j < nCols[i]; ++j)
                            {
                                fs << fmt.coeffSeparator;
                                fs << *ePointer++;
                            }
                            fs << fmt.rowSuffix;
                            fs << fmt.rowSeparator;
                        }
                        fs << rowSeparator;
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
            else
            {
                if (!FixedWidth)
                {
                    Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

                    for (int i = 0; i < nRows; i++)
                    {
                        TD *ePointer = idata[i];
                        for (int e = 0; e < entries; e++)
                        {
                            for (int j = 0; j < nCols[i]; j++)
                            {
                                std::stringstream sstr;
                                sstr.copyfmt(fs);
                                sstr << *ePointer++;
                                Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
                            }
                        }
                    }
                }

                if (Width)
                {
                    TD *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = idata[i] + e * nCols[i];

                            fs << form[i].rowPrefix;
                            fs.width(Width);
                            fs << *ePointer++;
                            for (int j = 1; j < nCols[i]; ++j)
                            {
                                fs << form[i].coeffSeparator;
                                fs.width(Width);
                                fs << *ePointer++;
                            }
                            fs << form[i].rowSuffix;
                            fs << form[i].rowSeparator;
                        }
                    }
                }
                else
                {
                    TD *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = idata[i] + e * nCols[i];

                            fs << form[i].rowPrefix;
                            fs << *ePointer++;
                            for (int j = 1; j < nCols[i]; ++j)
                            {
                                fs << form[i].coeffSeparator;
                                fs << *ePointer++;
                            }
                            fs << form[i].rowSuffix;
                            fs << form[i].rowSeparator;
                        }
                    }
                }

                return true;
            }
        }
    }

    /*!
     * \brief Helper function to save the matrix or a vector into a file 
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
    bool saveMatrix(TD *idata, int const nRows, int const nCols = 1, int const options = 0)
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

        setPrecision<TD>(fs);

        if (!FixedWidth)
        {
            Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

            for (int i = 0; i < nRows * nCols; i++)
            {
                std::stringstream sstr;
                sstr.copyfmt(fs);
                sstr << idata[i];
                Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
            }
        }

        if (Width)
        {
            if (nCols == 1)
            {
                fs << fmt.rowPrefix;
                fs.width(Width);
                fs << idata[0];
                for (int i = 1; i < nRows; i++)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << idata[i];
                }
                fs << fmt.rowSuffix;
                fs << fmt.rowSeparator;
            }
            else
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    fs << fmt.rowPrefix;
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
                    fs << fmt.rowSuffix;
                    fs << fmt.rowSeparator;
                }
            }
        }
        else
        {
            if (nCols == 1)
            {
                fs << fmt.rowPrefix;
                fs << idata[0];
                for (int i = 1; i < nRows; i++)
                {
                    fs << fmt.coeffSeparator;
                    fs << idata[i];
                }
                fs << fmt.rowSuffix;
                fs << fmt.rowSeparator;
            }
            else
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    fs << fmt.rowPrefix;
                    fs << idata[l];
                    for (int j = 1; j < nCols; j++)
                    {
                        l++;
                        fs << fmt.coeffSeparator;
                        fs << idata[l];
                    }
                    l++;
                    fs << fmt.rowSuffix;
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

    template <typename TD>
    bool saveMatrix(std::unique_ptr<TD[]> const &idata, int const nRows, int const nCols = 1, int const options = 0)
    {
        return saveMatrix<TD>(idata.get(), nRows, nCols, options);
    }

    /*!
     * \brief Helper function to save two arrays of data into a file 
     * 
     * \tparam TD          Data type 
     * 
     * \param idata        Array of input data of type TD
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type TD
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows in each data set
     */
    template <typename TD>
    bool saveMatrix(TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows)
    {
        if (!fs.is_open())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "This file stream is not open for writing." << std::endl;
            return false;
        }

        setPrecision<TD>(fs);

        if (!FixedWidth)
        {
            Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

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
        }

        if (Width)
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                fs << fmt.rowPrefix;
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
                fs << fmt.rowSuffix;
                fs << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                fs << fmt.rowPrefix;
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
                fs << fmt.rowSuffix;
                fs << fmt.rowSeparator;
            }
        }

        return true;
    }

    template <typename TD>
    bool saveMatrix(std::unique_ptr<TD[]> const &idata, int const idataCols, std::unique_ptr<TD[]> const &ifvalue, int const ifvalueCols, int const nRows)
    {
        return saveMatrix<TD>(idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows);
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
    bool loadMatrix(TD **idata, int const nRows, int const nCols, int const options = 0)
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
     * \brief Helper function to load one matrix (or entries number of matrcies) from a file 
     * 
     * \tparam  TD     Data type 
     * 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns for each row
     * \param options  (Default) 0 load matrix from matrix format and 1 load matrix from vector format
     * \param  entries Number of data entry
     *
     * \returns true if no error occurs during reading data
     */
    template <typename TD>
    bool loadMatrix(TD **idata, int const nRows, int const *nCols, int const options = 0, int const entries = 1)
    {
        std::string Line;

        //Default case
        if (entries == 1)
        {
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
        else
        {
            if (options == 0)
            {
                TD *ePointer;
                for (int e = 0; e < entries; e++)
                {
                    for (int i = 0; i < nRows; i++)
                    {
                        if (std::getline(fs, Line))
                        {
                            std::stringstream inLine(Line);

                            ePointer = idata[i] + e * nCols[i];

                            for (int j = 0; j < nCols[i]; j++)
                            {
                                inLine >> ePointer[j];
                            }
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
            else if (options == 1)
            {
                TD *ePointer;
                for (int e = 0; e < entries; e++)
                {
                    if (std::getline(fs, Line))
                    {
                        std::stringstream inLine(Line);

                        for (int i = 0; i < nRows; i++)
                        {
                            ePointer = idata[i] + e * nCols[i];

                            for (int j = 0; j < nCols[i]; j++)
                            {
                                inLine >> ePointer[j];
                            }
                        }
                    }
                    else
                    {
                        return false;
                    }
                    return true;
                }
            }
            return false;
        }
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
    bool loadMatrix(TD *idata, int const nRows, int const nCols = 1)
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

    template <typename TD>
    bool loadMatrix(std::unique_ptr<TD[]> &idata, int const nRows, int const nCols = 1)
    {
        return loadMatrix<TD>(idata.get(), nRows, nCols);
    }

    /*!
     * \brief Helper function to load two vector of data from a file
     * 
     * \tparam TD          Data type 
     * 
     * \param idata        Input array of data of type TD
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type TD
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows in each data set
     * 
     * \returns true if no error occurs during reading data
     */
    template <typename TD>
    bool loadMatrix(TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows)
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

    template <typename TD>
    bool loadMatrix(std::unique_ptr<TD[]> &idata, int const idataCols, std::unique_ptr<TD[]> &ifvalue, int const ifvalueCols, int const nRows)
    {
        return loadMatrix<TD>(idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows);
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD         Data type
     * 
     * \param  title       Title (string) that should be written at the top 
     * \param  idata       Array of input data of type TD
     * \param  nRows       Number of rows
     * \param  nCols       Number of columns
     * \param  printPrefix Prefix and suffix of the print  
     */
    template <typename TD>
    void printMatrix(const char *title, TD **idata, int const nRows, int const nCols, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        std::cout << printPrefix;
        if (std::strlen(title) > 0)
        {
            std::cout << title << "\n\n";
            if (!FixedWidth)
            {
                Width = 0;
            }
        }

        setPrecision<TD>(std::cout);

        if (!FixedWidth)
        {
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
        }

        if (Width)
        {
            for (int i = 0; i < nRows; ++i)
            {
                std::cout << fmt.rowPrefix;
                std::cout.width(Width);
                std::cout << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << idata[i][j];
                }
                std::cout << fmt.rowSuffix;
                std::cout << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0; i < nRows; ++i)
            {
                std::cout << fmt.rowPrefix;
                std::cout << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout << idata[i][j];
                }
                std::cout << fmt.rowSuffix;
                std::cout << fmt.rowSeparator;
            }
        }
        std::cout << printPrefix;
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam TD          Data type
     * 
     * \param idata        Array of input data of type TD
     * \param nRows        Number of rows
     * \param nCols        Number of columns
     * \param  printPrefix Prefix and suffix of the print  
     */
    template <typename TD>
    void printMatrix(TD **idata, int const nRows, int const nCols, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        printMatrix<TD>("", idata, nRows, nCols, printPrefix);
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD     Data type
     * 
     * \param   idata  Array of input data of type TD
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns
     * \param   form   Print format
     */
    template <typename TD>
    void printMatrix(TD **idata, int const nRows, int const nCols, ioFormat const &form)
    {
        setPrecision<TD>(std::cout);

        if (!FixedWidth)
        {
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
        }

        if (Width)
        {
            for (int i = 0; i < nRows; ++i)
            {
                std::cout << form.rowPrefix;
                std::cout.width(Width);
                std::cout << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    std::cout << form.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << idata[i][j];
                }
                std::cout << form.rowSuffix;
                std::cout << form.rowSeparator;
            }
        }
        else
        {
            for (int i = 0; i < nRows; ++i)
            {
                std::cout << form.rowPrefix;
                std::cout << idata[i][0];
                for (int j = 1; j < nCols; ++j)
                {
                    std::cout << form.coeffSeparator;
                    std::cout << idata[i][j];
                }
                std::cout << form.rowSuffix;
                std::cout << form.rowSeparator;
            }
        }
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD     Data type
     * 
     * \param  title       Title (string) that should be written at the top 
     * \param  idata       Array of input data of type TD
     * \param  nRows       Number of rows
     * \param  nCols       Number of columns for each row
     * \param  entries     Number of data entry   
     * \param  printPrefix Prefix and suffix of the print  
     */
    template <typename TD>
    void printMatrix(const char *title, TD **idata, int const nRows, int const *nCols, int const entries = 1, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        std::cout << printPrefix;
        if (std::strlen(title) > 0)
        {
            std::cout << title << "\n\n";
            if (!FixedWidth)
            {
                Width = 0;
            }
        }

        setPrecision<TD>(std::cout);

        //DEfault case one set of data
        if (entries == 1)
        {
            if (!FixedWidth)
            {
                for (int i = 0; i < nRows; i++)
                {
                    for (int j = 0; j < nCols[i]; j++)
                    {
                        std::stringstream sstr;
                        sstr.copyfmt(std::cout);
                        sstr << idata[i][j];
                        Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
                    }
                }
            }

            if (Width)
            {
                for (int i = 0; i < nRows; ++i)
                {
                    std::cout << fmt.rowPrefix;
                    std::cout.width(Width);
                    std::cout << idata[i][0];
                    for (int j = 1; j < nCols[i]; ++j)
                    {
                        std::cout << fmt.coeffSeparator;
                        std::cout.width(Width);
                        std::cout << idata[i][j];
                    }
                    std::cout << fmt.rowSuffix;
                    std::cout << fmt.rowSeparator;
                }
            }
            else
            {
                for (int i = 0; i < nRows; ++i)
                {
                    std::cout << fmt.rowPrefix;
                    std::cout << idata[i][0];
                    for (int j = 1; j < nCols[i]; ++j)
                    {
                        std::cout << fmt.coeffSeparator;
                        std::cout << idata[i][j];
                    }
                    std::cout << fmt.rowSuffix;
                    std::cout << fmt.rowSeparator;
                }
            }
        }
        else
        {
            if (!FixedWidth)
            {
                for (int i = 0; i < nRows; i++)
                {
                    TD *ePointer = idata[i];
                    for (int e = 0; e < entries; e++)
                    {
                        for (int j = 0; j < nCols[i]; j++)
                        {
                            std::stringstream sstr;
                            sstr.copyfmt(std::cout);
                            sstr << *ePointer++;
                            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
                        }
                    }
                }
            }

            if (Width)
            {
                TD *ePointer;
                for (int e = 0; e < entries; e++)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        ePointer = idata[i] + e * nCols[i];

                        std::cout << fmt.rowPrefix;
                        std::cout.width(Width);
                        std::cout << *ePointer++;
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            std::cout << fmt.coeffSeparator;
                            std::cout.width(Width);
                            std::cout << *ePointer++;
                        }
                        std::cout << fmt.rowSuffix;
                        std::cout << fmt.rowSeparator;
                    }
                }
            }
            else
            {
                TD *ePointer;
                for (int e = 0; e < entries; e++)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        ePointer = idata[i] + e * nCols[i];

                        std::cout << fmt.rowPrefix;
                        std::cout << *ePointer++;
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            std::cout << fmt.coeffSeparator;
                            std::cout << *ePointer++;
                        }
                        std::cout << fmt.rowSuffix;
                        std::cout << fmt.rowSeparator;
                    }
                }
            }
        }
        std::cout << printPrefix;
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam TD          Data type
     * 
     * \param  idata       Array of input data of type TD
     * \param  nRows       Number of rows
     * \param  nCols       Number of columns for each row
     * \param  printPrefix Prefix and suffix of the print  
     */
    template <typename TD>
    void printMatrix(TD **idata, int const nRows, int const *nCols, int const entries = 1, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        printMatrix<TD>("", idata, nRows, nCols, entries, printPrefix);
    }

    /*!
     * \brief Helper function to print one matrix (or @entries number of matrices)
     * 
     * \tparam  TD     Data type
     * 
     * \param  title   Title (string) that should be written at the top 
     * \param  idata   Array of input data of type TD
     * \param  nRows   Number of rows
     * \param  nCols   Number of columns for each row
     * \param  entries Number of data entry   
     * \param  form    Print format for each row 
     */
    template <typename TD>
    void printMatrix(TD **idata, int const nRows, int const *nCols, int const entries = 1, std::vector<ioFormat> const &form = std::vector<ioFormat>())
    {
        if (form.size() != nRows)
        {
            printMatrix<TD>(idata, nRows, nCols, entries);
        }
        else
        {
            setPrecision<TD>(std::cout);

            //Default case
            if (entries == 1)
            {
                if (!FixedWidth)
                {
                    for (int i = 0; i < nRows; i++)
                    {
                        for (int j = 0; j < nCols[i]; j++)
                        {
                            std::stringstream sstr;
                            sstr.copyfmt(std::cout);
                            sstr << idata[i][j];
                            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
                        }
                    }
                }

                if (Width)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        std::cout << form[i].rowPrefix;
                        std::cout.width(Width);
                        std::cout << idata[i][0];
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            std::cout << form[i].coeffSeparator;
                            std::cout.width(Width);
                            std::cout << idata[i][j];
                        }
                        std::cout << form[i].rowSuffix;
                        std::cout << form[i].rowSeparator;
                    }
                }
                else
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        std::cout << form[i].rowPrefix;
                        std::cout << idata[i][0];
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            std::cout << form[i].coeffSeparator;
                            std::cout << idata[i][j];
                        }
                        std::cout << form[i].rowSuffix;
                        std::cout << form[i].rowSeparator;
                    }
                }
            }
            else
            {
                if (!FixedWidth)
                {
                    for (int i = 0; i < nRows; i++)
                    {
                        TD *ePointer = idata[i];
                        for (int e = 0; e < entries; e++)
                        {
                            for (int j = 0; j < nCols[i]; j++)
                            {
                                std::stringstream sstr;
                                sstr.copyfmt(std::cout);
                                sstr << *ePointer++;
                                Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
                            }
                        }
                    }
                }

                if (Width)
                {
                    TD *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = idata[i] + e * nCols[i];

                            std::cout << form[i].rowPrefix;
                            std::cout.width(Width);
                            std::cout << *ePointer++;
                            for (int j = 1; j < nCols[i]; ++j)
                            {
                                std::cout << form[i].coeffSeparator;
                                std::cout.width(Width);
                                std::cout << *ePointer++;
                            }
                            std::cout << form[i].rowSuffix;
                            std::cout << form[i].rowSeparator;
                        }
                    }
                }
                else
                {
                    TD *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = idata[i] + e * nCols[i];

                            std::cout << form[i].rowPrefix;
                            std::cout << *ePointer++;
                            for (int j = 1; j < nCols[i]; ++j)
                            {
                                std::cout << form[i].coeffSeparator;
                                std::cout << *ePointer++;
                            }
                            std::cout << form[i].rowSuffix;
                            std::cout << form[i].rowSeparator;
                        }
                    }
                }
            }
        }
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD         Data type
     * 
     * \param  title       Title (string) that should be written at the top 
     * \param  idata       Array of input data of type TD
     * \param  nRows       Number of rows
     * \param  nCols       Number of columns (default is 1)
     * \param  printPrefix Prefix and suffix of the print  
     */
    template <typename TD>
    void printMatrix(const char *title, TD *idata, int const nRows, int const nCols = 1, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        std::cout << printPrefix;
        if (std::strlen(title) > 0)
        {
            std::cout << title << "\n\n";

            if (!FixedWidth)
            {
                Width = 0;
            }
        }

        setPrecision<TD>(std::cout);

        if (!FixedWidth)
        {
            for (int i = 0; i < nRows * nCols; i++)
            {
                std::stringstream sstr;
                sstr.copyfmt(std::cout);
                sstr << idata[i];
                Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
            }
        }

        if (nCols == 1)
        {
            std::cout << fmt.rowPrefix;
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
            std::cout << fmt.rowSuffix;
            std::cout << fmt.rowSeparator;
        }
        else
        {
            if (Width)
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    std::cout << fmt.rowPrefix;
                    std::cout.width(Width);
                    std::cout << idata[l];
                    for (int j = 1; j < nCols; j++, l++)
                    {
                        std::cout << fmt.coeffSeparator;
                        std::cout.width(Width);
                        std::cout << idata[l];
                    }
                    std::cout << fmt.rowSuffix;
                    std::cout << fmt.rowSeparator;
                }
            }
            else
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    std::cout << fmt.rowPrefix;
                    std::cout << idata[l];
                    for (int j = 1; j < nCols; j++, l++)
                    {
                        std::cout << fmt.coeffSeparator;
                        std::cout << idata[l];
                    }
                    std::cout << fmt.rowSuffix;
                    std::cout << fmt.rowSeparator;
                }
            }
        }
        std::cout << printPrefix;
    }

    template <typename TD>
    void printMatrix(const char *title, std::unique_ptr<TD[]> const &idata, int const nRows, int const nCols = 1, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        printMatrix<TD>(title, idata.get(), nRows, nCols, printPrefix);
    }

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam  TD    Data type
     * 
     * \param  idata  Array of input data of type TD
     * \param  nRows  Number of rows
     * \param  nCols  Number of columns (default is 1)
     * \param  form   Print format
     */
    template <typename TD>
    void printMatrix(TD *idata, int const nRows, int const nCols = 1, ioFormat const &form = ioFormat("NO"))
    {
        if (form.coeffSeparator == "NO")
        {
            printMatrix<TD>("", idata, nRows, nCols);
        }
        else
        {
            setPrecision<TD>(std::cout);

            if (!FixedWidth)
            {
                for (int i = 0; i < nRows * nCols; i++)
                {
                    std::stringstream sstr;
                    sstr.copyfmt(std::cout);
                    sstr << idata[i];
                    Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
                }
            }

            if (nCols == 1)
            {
                std::cout << form.rowPrefix;
                if (Width)
                {
                    std::cout.width(Width);
                    std::cout << idata[0];
                    for (int i = 1; i < nRows; i++)
                    {
                        std::cout << form.coeffSeparator;
                        std::cout.width(Width);
                        std::cout << idata[i];
                    }
                }
                else
                {
                    std::cout << idata[0];
                    for (int i = 1; i < nRows; i++)
                    {
                        std::cout << form.coeffSeparator;
                        std::cout << idata[i];
                    }
                }
                std::cout << form.rowSuffix;
                std::cout << form.rowSeparator;
            }
            else
            {
                if (Width)
                {
                    for (int i = 0, l = 0; i < nRows; i++)
                    {
                        std::cout << form.rowPrefix;
                        std::cout.width(Width);
                        std::cout << idata[l];
                        for (int j = 1; j < nCols; j++, l++)
                        {
                            std::cout << form.coeffSeparator;
                            std::cout.width(Width);
                            std::cout << idata[l];
                        }
                        std::cout << form.rowSuffix;
                        std::cout << form.rowSeparator;
                    }
                }
                else
                {
                    for (int i = 0, l = 0; i < nRows; i++)
                    {
                        std::cout << form.rowPrefix;
                        std::cout << idata[l];
                        for (int j = 1; j < nCols; j++, l++)
                        {
                            std::cout << form.coeffSeparator;
                            std::cout << idata[l];
                        }
                        std::cout << form.rowSuffix;
                        std::cout << form.rowSeparator;
                    }
                }
            }
        }
    }

    template <typename TD>
    void printMatrix(std::unique_ptr<TD[]> const &idata, int const nRows, int const nCols = 1, ioFormat const &form = ioFormat("NO"))
    {
        printMatrix<TD>(idata.get(), nRows, nCols, form);
    }

    /*!
     * \brief Helper function to print one element of input data
     * 
     * \tparam  TD    Data type
     * 
     * \param  idata  Array of input data of type TD
     * \param  form   Print format
     */
    template <typename TD>
    void printMatrix(TD *idata, ioFormat const &form)
    {
        setPrecision<TD>(std::cout);

        if (!FixedWidth)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << *idata;
            Width = std::max<std::ptrdiff_t>(Width, Idx(sstr.str().length()));
        }

        std::cout << form.rowPrefix;
        if (Width)
        {
            std::cout.width(Width);
        }
        std::cout << *idata;
        std::cout << form.rowSuffix;
        std::cout << form.rowSeparator;
    }

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam TD             Data type
     * 
     * \param  title          Title (string) that should be written at the top 
     * \param  idata          Array of input data of type TD
     * \param  idataCols      Number of columns of inpput array data (idata)
     * \param  ifvalue        Array of input value data of type TD
     * \param  ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param  nRows          Number of rows
     * \param  printPrefix    Prefix and suffix of the print  
     */
    template <typename TD>
    void printMatrix(const char *title, TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        std::cout << printPrefix;
        if (std::strlen(title) > 0)
        {
            std::cout << title << "\n\n";

            if (!FixedWidth)
            {
                Width = 0;
            }
        }

        setPrecision<TD>(std::cout);

        if (!FixedWidth)
        {
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
        }

        if (Width)
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                std::cout << fmt.rowPrefix;
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
                std::cout << fmt.rowSuffix;
                std::cout << fmt.rowSeparator;
            }
        }
        else
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                std::cout << fmt.rowPrefix;
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
                std::cout << fmt.rowSuffix;
                std::cout << fmt.rowSeparator;
            }
        }
        std::cout << printPrefix;
    }

    template <typename TD>
    void printMatrix(const char *title, std::unique_ptr<TD[]> const &idata, int const idataCols, std::unique_ptr<TD[]> const &ifvalue, int const ifvalueCols, int const nRows, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        printMatrix<TD>(title, idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows, printPrefix);
    }

    template <typename TD>
    void printMatrix(TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        printMatrix<TD>("", idata, idataCols, ifvalue, ifvalueCols, nRows, printPrefix);
    }

    template <typename TD>
    void printMatrix(std::unique_ptr<TD[]> const &idata, int const idataCols, std::unique_ptr<TD[]> const &ifvalue, int const ifvalueCols, int const nRows, std::string const &printPrefix = "\n----------------------------------------\n")
    {
        printMatrix<TD>("", idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows, printPrefix);
    }

    /*!
     * \brief Helper function to print two vectors of data with stream format for each
     * 
     * \tparam TD             Data type
     * 
     * \param  title          Title (string) that should be written at the top 
     * \param  idata          Array of input data of type TD
     * \param  idataCols      Number of columns of inpput array data (idata)
     * \param  ifvalue        Array of input value data of type TD
     * \param  ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param  nRows          Number of rows
     * \param  formD          Print format for input data
     * \param  formF          Print format for input function value 
     */
    template <typename TD>
    void printMatrix(TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows, ioFormat const &formD, ioFormat const &formF)
    {
        setPrecision<TD>(std::cout);

        if (!FixedWidth)
        {
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
        }

        if (Width)
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                std::cout << formD.rowPrefix;
                std::cout.width(Width);
                std::cout << idata[l++];
                for (int j = 1; j < idataCols; j++)
                {
                    std::cout << formD.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << idata[l++];
                }
                std::cout << formD.rowSuffix;
                std::cout << formD.rowSeparator;
                std::cout << formF.rowPrefix;
                for (int j = 0; j < ifvalueCols; j++)
                {
                    std::cout << formF.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << ifvalue[k++];
                }
                std::cout << formF.rowSuffix;
                std::cout << formF.rowSeparator;
            }
        }
        else
        {
            for (int i = 0, l = 0, k = 0; i < nRows; i++)
            {
                std::cout << formD.rowPrefix;
                std::cout << idata[l++];
                for (int j = 1; j < idataCols; j++)
                {
                    std::cout << formD.coeffSeparator;
                    std::cout << idata[l++];
                }
                std::cout << formD.rowSuffix;
                std::cout << formD.rowSeparator;
                std::cout << formF.rowPrefix;
                for (int j = 0; j < ifvalueCols; j++)
                {
                    std::cout << formF.coeffSeparator;
                    std::cout << ifvalue[k++];
                }
                std::cout << formF.rowSuffix;
                std::cout << formF.rowSeparator;
            }
        }
    }

    template <typename TD>
    void printMatrix(std::unique_ptr<TD[]> const &idata, int const idataCols, std::unique_ptr<TD[]> const &ifvalue, int const ifvalueCols, int const nRows, ioFormat const &formD, ioFormat const &formF)
    {
        printMatrix<TD>(idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows, formD, formF);
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

    //! Flag for using the pre defined Width or computing it on the fly from the input data
    bool FixedWidth;

    //! IO format
    ioFormat fmt;
};

#endif
