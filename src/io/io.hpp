#ifndef UMUQ_IO_H
#define UMUQ_IO_H

namespace umuq
{

/*! \defgroup IO_Module IO module
 * This is the Input/OUTPUT module of UMUQ providing all necessary classes of reading and writing data.
 */

/*! \class ioFormat
 * \ingroup IO_Module
 * 
 * \brief Stores a set of parameters controlling the way matrices are printed
 *
 * Controlling the way matrices are printed. <br>
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
     * \brief Construct a new ioFormat object
     * 
     * \param CoeffSeparator  String printed between two coefficients of the same row
     * \param RowSeparator    String printed between two rows
     * \param RowPrefix       String printed at the beginning of each row
     * \param RowSuffix       String printed at the end of each row
     */
    ioFormat(std::string const &CoeffSeparator = " ",
             std::string const &RowSeparator = "\n",
             std::string const &RowPrefix = "",
             std::string const &RowSuffix = "");

    /*!
     * \brief Operator \c == compares the underlying ioFormat object
     * 
     * \param rhs ioFormat object
     * 
     * \returns true If lhs \c == rhs 
     */
    inline bool operator==(ioFormat const &rhs);

    /*!
     * \brief Operator \c == compares the underlying ioFormat object
     * 
     * \param rhs ioFormat object
     * 
     * \returns true If lhs \c == rhs 
     */
    inline bool operator==(ioFormat const &rhs) const;

    /*!
     * \brief Operator \c != compares the underlying ioFormat object
     * 
     * \param rhs  ioFormat object
     * 
     * \returns true  If lhs \c != rhs 
     */
    inline bool operator!=(ioFormat const &rhs);

    /*!
     * \brief Operator \c != compares the underlying ioFormat object
     * 
     * \param rhs  ioFormat object
     * 
     * \returns true  If lhs \c != rhs 
     */
    inline bool operator!=(ioFormat const &rhs) const;

  public:
    /*! String printed between two coefficients */
    std::string coeffSeparator;
    /*! String printed between two rows */
    std::string rowSeparator;
    /*! String printed at the beginning of each row */
    std::string rowPrefix;
    /*! String printed at the end of each row */
    std::string rowSuffix;
};

/*!
 * \todo
 * All the functions should use the ioFormat for printing, save and load and
 * we should remove all options from them.
 */

/*! \class io
 * \ingroup IO_Module
 * 
 * \brief This class includes IO functionality.
 *
 * List of available functions: <br>
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
    /*! \var static const std::ios_base::openmode app
     * \brief Seeks to the end of stream before each write
     */
    static const std::ios_base::openmode app = std::fstream::app;
    /*! \var static const std::ios_base::openmode binary
     * \brief Binary mode
     */
    static const std::ios_base::openmode binary = std::fstream::binary;
    /*! \var static const std::ios_base::openmode in
     * \brief Reading mode
     */
    static const std::ios_base::openmode in = std::fstream::in;
    /*! \var static const std::ios_base::openmode out
     * \brief Writing mode
     */
    static const std::ios_base::openmode out = std::fstream::out;
    /*! \var static const std::ios_base::openmode trunc
     * \brief Discard the contents of the stream when opening
     */
    static const std::ios_base::openmode trunc = std::fstream::trunc;
    /*! \var static const std::ios_base::openmode ate
     * \brief Seeks to the end of stream immediately after open
     */
    static const std::ios_base::openmode ate = std::fstream::ate;

    /*!
     * \brief Construct a new io object
     * 
     */
    io();

    /*!
     * \brief Destroy the io object
     * 
     */
    ~io();

    /*!
     * \brief Check to see whether the file is opened or not
     * 
     * \returns true If the file is already opened 
     */
    inline bool isFileOpened() const;

    /*!
     * \brief Check to see whether the file fileName exists and accessible to read or write!
     *  
     * \returns true If the file exists 
     */
    inline bool isFileExist(const char *fileName);

    /*!
     * \brief Check to see whether the file fileName exists and accessible to read or write!
     *  
     * \returns true If the file exists 
     */
    inline bool isFileExist(std::string const &fileName);

    /*!
     * \brief Opens the file whose name is specified with the parameter filename 
     *  
     * Opens the file whose name is specified in the parameter filename and
     * associates it with a stream that can be identified in future operations 
     * by the FILE pointer returned.   
     * 
     * Available file open flags: <br>
     * - \b std::fstream::app 	  Seeks to the end of stream before each write
     * - \b std::fstream::binary  Open in binary mode
     * - \b std::fstream::in 	  Open for reading
     * - \b std::fstream::out 	  Open for writing
     * - \b std::fstream::trunc   Discard the contents of the stream when opening
     * - \b std::fstream::ate 	  Seeks to the end of stream immediately after open
     * 
     * \returns true If everything goes OK
     */
    bool openFile(const char *fileName, const std::ios_base::openmode mode = in);

    /*!
     * \brief Opens the file whose name is specified with the parameter filename 
     *  
     * Opens the file whose name is specified in the parameter filename and
     * associates it with a stream that can be identified in future operations 
     * by the FILE pointer returned.   
     * 
     * Available file open flags: <br>
     * - \b std::fstream::app 	  Seeks to the end of stream before each write
     * - \b std::fstream::binary  Open in binary mode
     * - \b std::fstream::in 	  Open for reading
     * - \b std::fstream::out 	  Open for writing
     * - \b std::fstream::trunc   Discard the contents of the stream when opening
     * - \b std::fstream::ate 	  Seeks to the end of stream immediately after open
     * 
     * \returns true If everything goes OK
     */
    bool openFile(std::string const &fileName, const std::ios_base::openmode mode = in);

    /*!
     * \brief Get string from stream
     * 
     * Get a string from stream and stores them into line until 
     * a newline or the end-of-file is reached, whichever happens first.
     * 
     * \returns true If no error occurs on the associated stream
     */
    bool readLine(const char comment = '#');

    /*!
     * \brief Set position of stream at the beginning
     * 
     * Sets the position indicator associated with stream to the beginning of the file.
     */
    inline void rewindFile();

    /*!
     * \brief Close the File
     * 
     */
    inline void closeFile();

    /*!
     * \brief Get the stream
     * 
     */
    inline std::fstream &getFstream();

    /*!
     * \brief Get the Line object
     * 
     * \returns std::string& 
     */
    inline std::string &getLine();

    /*!
     * \brief Set the stream Precision 
     * 
     * \tparam DataType Data type
     * 
     * \param streamBuffer  Stream buffer to use as output sequence
     */
    template <typename DataType>
    inline void setPrecision(std::ostream &streamBuffer);

    /*!
     * \brief Set the width parameter of the stream to exactly n.
     * 
     * If Input streamWidth is \c < 0, the function will set the stream to zero and its setting flag to false 
     * 
     * \param streamWidth Value for Width parameter of the stream  
     */
    inline void setWidth(int const streamWidth = 0);

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints    Input data points
     * \param nRows         Number of Rows
     * \param nCols         Number of Columns
     * \param streamBuffer  Stream buffer to use as output sequence 
     * 
     * \returns The width parameter of the stream
     */
    template <typename DataType>
    int getWidth(DataType const *dataPoints, int const nRows, int const nCols, std::ostream &streamBuffer);

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints    Input data points
     * \param nRows         Number of Rows
     * \param nCols         Number of Columns
     * \param streamBuffer  Stream buffer to use as output sequence 
     * 
     * \returns The width parameter of the stream
     */
    template <typename DataType>
    int getWidth(std::unique_ptr<DataType[]> const &dataPoints, int const nRows, int const nCols, std::ostream &streamBuffer);

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints    Input data points
     * \param nRows         Number of Rows
     * \param nCols         Number of Columns
     * \param streamBuffer  Stream buffer to use as output sequence 
     * 
     * \returns The width parameter of the stream
     */
    template <typename DataType>
    int getWidth(std::vector<DataType> const &dataPoints, int const nRows, int const nCols, std::ostream &streamBuffer);

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam DataType data type
     * 
     * \param dataPoints    Input data points
     * \param nRows         Number of Rows
     * \param nCols         Number of Columns
     * \param streamBuffer  Stream buffer to use as output sequence 
     * 
     * \returns The width parameter of the stream
     */
    template <typename DataType>
    int getWidth(DataType **dataPoints, int const nRows, int const nCols, std::ostream &streamBuffer);

    /*!
     * \brief Helper function to save the matrix of type EigenMatrixType with IOFormatType format into a file 
     * 
     * \tparam EigenMatrixType Eigen matrix type 
     * \tparam IOFormatType IO format type (We can use either of \c ioFormat or \c Eigen::IOFormat)
     * 
     * \param dataMatrix   Input data points matrix
     * \param ioformat     IO format for the matrix type
     *
     * \returns true If no error occurs during writing the matrix
     */
    template <typename EigenMatrixType, typename IOFormatType>
    bool saveMatrix(EigenMatrixType &dataMatrix, IOFormatType const &ioformat);

    /*!
     * \brief Helper function to save one matrix into a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param   dataPoints  Input data points
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns
     * \param options  (0 Default) 
     *                  - \b 0 Save matrix in matrix format and proceed the position indicator to the next line & 
     *                  - \b 1 Save matrix in vector format and proceed the position indicator to the next line &
     *                  - \b 2 Save matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename DataType>
    bool saveMatrix(DataType **dataPoints, int const nRows, int const nCols, int const options = 0);

    /*!
     * \brief Helper function to save one matrix (or entries number matrices) into a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param   nRows     Number of rows
     * \param   nCols     Number of columns for each row
     * \param options     (0 Default) 
     *                      - \b 0 Saves matrix in matrix format and proceeds the position indicator to the next line &  
     *                      - \b 1 Saves matrix in vector format and proceeds the position indicator to the next line & 
     *                      - \b 2 Saves matrix in vector format and keep the position indicator on the same line
     * \param  entries    Number of data entry (Input data contains pointer to array of data)
     * \param  ioformat   Print format for each row 
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename DataType>
    bool saveMatrix(DataType **dataPoints,
                    int const nRows,
                    int const *nCols,
                    int const options = 0,
                    int const entries = 1,
                    std::vector<ioFormat> const &ioformat = EmptyVector<ioFormat>);

    /*!
     * \brief Helper function to save the matrix or a vector into a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows 
     * \param nCols       Number of columns for each row (default is 1)
     * \param options     (0 Default) 
     *                      - \b  0 Saves matrix in matrix format and proceeds the position indicator to the next line &  
     *                      - \b 1 Saves matrix in vector format and proceeds the position indicator to the next line & 
     *                      - \b 2 Saves matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename DataType>
    bool saveMatrix(DataType const *dataPoints,
                    int const nRows,
                    int const nCols = 1,
                    int const options = 0);

    /*!
     * \brief Helper function to save the matrix or a vector into a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows 
     * \param nCols       Number of columns for each row (default is 1)
     * \param options     (0 Default) 
     *                      - \b  0 Saves matrix in matrix format and proceeds the position indicator to the next line &  
     *                      - \b 1 Saves matrix in vector format and proceeds the position indicator to the next line & 
     *                      - \b 2 Saves matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename DataType>
    bool saveMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                    int const nRows,
                    int const nCols = 1,
                    int const options = 0);

    /*!
     * \brief Helper function to save the matrix or a vector into a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows 
     * \param nCols       Number of columns for each row (default is 1)
     * \param options     (0 Default) 
     *                      - \b  0 Saves matrix in matrix format and proceeds the position indicator to the next line &  
     *                      - \b 1 Saves matrix in vector format and proceeds the position indicator to the next line & 
     *                      - \b 2 Saves matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename DataType>
    bool saveMatrix(std::vector<DataType> const &dataPoints,
                    int const nRows,
                    int const nCols = 1,
                    int const options = 0);

    /*!
     * \brief Helper function to save two arrays of data into a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows in each data set
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename DataType>
    bool saveMatrix(DataType const *dataPoints,
                    int const dataPointsDim,
                    DataType const *functionValues,
                    int const nFunctionValues,
                    int const nRows);

    /*!
     * \brief Helper function to save two arrays of data into a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows in each data set
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename DataType>
    bool saveMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                    int const dataPointsDim,
                    std::unique_ptr<DataType[]> const &functionValues,
                    int const nFunctionValues,
                    int const nRows);

    /*!
     * \brief Helper function to save two arrays of data into a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows in each data set
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename DataType>
    bool saveMatrix(std::vector<DataType> const &dataPoints,
                    int const dataPointsDim,
                    std::vector<DataType> const &functionValues,
                    int const nFunctionValues,
                    int const nRows);

    /*!
     * \brief Helper function to load the matrix of type EigenMatrixType from a file 
     * 
     * \tparam EigenMatrixType Eigen Matrix type
     * 
     * \param dataMatrix  Input data points matrix
     *
     * \returns true If no error occurs during reading a matrix
     */
    template <typename EigenMatrixType>
    bool loadMatrix(EigenMatrixType &dataMatrix);

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns
     * \param options     (0 default) Load matrix from matrix format and 1 load matrix from vector format
     *
     * \returns true If no error occurs during reading data
     */
    template <typename DataType>
    bool loadMatrix(DataType **dataPoints, int const nRows, int const nCols, int const options = 0);

    /*!
     * \brief Helper function to load one matrix (or entries number of matrices) from a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns for each row
     * \param options     (0 Default) Load matrix from matrix format and 1 load matrix from vector format
     * \param entries     Number of data entry
     *
     * \returns true If no error occurs during reading data
     */
    template <typename DataType>
    bool loadMatrix(DataType **dataPoints, int const nRows, int const *nCols, int const options = 0, int const entries = 1);

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns for each row (default is 1)
     *
     * \returns true If no error occurs during reading data
     */
    template <typename DataType>
    bool loadMatrix(DataType *dataPoints, int const nRows, int const nCols = 1);

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns for each row (default is 1)
     *
     * \returns true If no error occurs during reading data
     */
    template <typename DataType>
    bool loadMatrix(std::unique_ptr<DataType[]> &dataPoints, int const nRows, int const nCols = 1);

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns for each row (default is 1)
     *
     * \returns true If no error occurs during reading data
     */
    template <typename DataType>
    bool loadMatrix(std::vector<DataType> &dataPoints, int const nRows, int const nCols = 1);

    /*!
     * \brief Helper function to load two vector of data from a file
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints        Input data points
     * \param dataPointsDim     Number of columns of input array data (data dimension)
     * \param functionValues    Array of input value data
     * \param nFunctionValues   Number of columns of input value data (functionValues at data points)
     * \param nRows             Number of rows in each data set
     * 
     * \returns true If no error occurs during reading data
     */
    template <typename DataType>
    bool loadMatrix(DataType *dataPoints, int const dataPointsDim, DataType *functionValues, int const nFunctionValues, int const nRows);

    /*!
     * \brief Helper function to load two vector of data from a file
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows in each data set
     * 
     * \returns true If no error occurs during reading data
     */
    template <typename DataType>
    bool loadMatrix(std::unique_ptr<DataType[]> &dataPoints, int const dataPointsDim, std::unique_ptr<DataType[]> &functionValues, int const nFunctionValues, int const nRows);

    /*!
     * \brief Helper function to load two vector of data from a file
     * 
     * \tparam DataType Data type 
     * 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows in each data set
     * 
     * \returns true If no error occurs during reading data
     */
    template <typename DataType>
    bool loadMatrix(std::vector<DataType> &dataPoints, int const dataPointsDim, std::vector<DataType> &functionValues, int const nFunctionValues, int const nRows);

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param title        Title (string) that should be written at the top 
     * \param dataPoints   Input data points
     * \param nRows        Number of rows
     * \param nCols        Number of columns
     * \param printPrefix  Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(const char *title,
                     DataType **dataPoints,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints   Input data points
     * \param nRows        Number of rows
     * \param nCols        Number of columns
     * \param printPrefix  Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(DataType **dataPoints,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns
     * \param ioformat    Print format
     */
    template <typename DataType>
    void printMatrix(DataType **dataPoints,
                     int const nRows,
                     int const nCols,
                     ioFormat const &ioformat);

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param title        Title (string) that should be written at the top 
     * \param dataPoints   Input data points
     * \param nRows        Number of rows
     * \param nCols        Number of columns for each row
     * \param entries      Number of data entry   
     * \param printPrefix  Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(const char *title,
                     DataType **dataPoints,
                     int const nRows,
                     int const *nCols,
                     int const entries = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints   Input data points
     * \param nRows        Number of rows
     * \param nCols        Number of columns for each row
     * \param printPrefix  Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(DataType **dataPoints,
                     int const nRows,
                     int const *nCols,
                     int const entries = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print one matrix (or entries number of matrices)
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns for each row
     * \param entries     Number of data entry   
     * \param ioformat    Print format for each row 
     */
    template <typename DataType>
    void printMatrix(DataType **dataPoints,
                     int const nRows,
                     int const *nCols,
                     int const entries = 1,
                     std::vector<ioFormat> const &ioformat = EmptyVector<ioFormat>);

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param title        Title (string) that should be written at the top 
     * \param dataPoints   Input data points
     * \param nRows        Number of rows
     * \param nCols        Number of columns (default is 1)
     * \param printPrefix  Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(const char *title,
                     DataType const *dataPoints,
                     int const nRows,
                     int const nCols = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param title        Title (string) that should be written at the top 
     * \param dataPoints   Input data points
     * \param nRows        Number of rows
     * \param nCols        Number of columns (default is 1)
     * \param printPrefix  Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(const char *title,
                     std::unique_ptr<DataType[]> const &dataPoints,
                     int const nRows,
                     int const nCols = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param title        Title (string) that should be written at the top 
     * \param dataPoints   Input data points
     * \param nRows        Number of rows
     * \param nCols        Number of columns (default is 1)
     * \param printPrefix  Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(const char *title,
                     std::vector<DataType> const &dataPoints,
                     int const nRows,
                     int const nCols = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns (default is 1)
     * \param ioformat    Print format
     */
    template <typename DataType>
    void printMatrix(DataType const *dataPoints,
                     int const nRows,
                     int const nCols = 1,
                     ioFormat const &ioformat = ioFormat("NO"));

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns (default is 1)
     * \param ioformat    Print format
     */
    template <typename DataType>
    void printMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                     int const nRows,
                     int const nCols = 1,
                     ioFormat const &ioformat = ioFormat("NO"));

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints  Input data points
     * \param nRows       Number of rows
     * \param nCols       Number of columns (default is 1)
     * \param ioformat    Print format
     */
    template <typename DataType>
    void printMatrix(std::vector<DataType> const &dataPoints,
                     int const nRows,
                     int const nCols = 1,
                     ioFormat const &ioformat = ioFormat("NO"));

    /*!
     * \brief Helper function to print one element of input data
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints  Input data points
     * \param ioformat    Print format
     */
    template <typename DataType>
    void printMatrix(DataType const *dataPoints, ioFormat const &ioformat);

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam DataType Data type
     * 
     * \param title            Title (string) that should be written at the top 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows
     * \param printPrefix      Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(const char *title,
                     DataType const *dataPoints,
                     int const dataPointsDim,
                     DataType const *functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam DataType Data type
     * 
     * \param title            Title (string) that should be written at the top 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows
     * \param printPrefix      Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(const char *title,
                     std::unique_ptr<DataType[]> const &dataPoints,
                     int const dataPointsDim,
                     std::unique_ptr<DataType[]> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam DataType Data type
     * 
     * \param title            Title (string) that should be written at the top 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows
     * \param printPrefix      Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(const char *title,
                     std::vector<DataType> const &dataPoints,
                     int const dataPointsDim,
                     std::vector<DataType> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam DataType Data type
     * 
     * \param title            Title (string) that should be written at the top 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows
     * \param printPrefix      Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(DataType const *dataPoints,
                     int const dataPointsDim, DataType *functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam DataType Data type
     * 
     * \param title            Title (string) that should be written at the top 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows
     * \param printPrefix      Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                     int const dataPointsDim,
                     std::unique_ptr<DataType[]> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam DataType Data type
     * 
     * \param title            Title (string) that should be written at the top 
     * \param dataPoints       Input data points
     * \param dataPointsDim    Number of columns of input array data (data dimension)
     * \param functionValues   Array of input value data
     * \param nFunctionValues  Number of columns of input value data (functionValues at data points)
     * \param nRows            Number of rows
     * \param printPrefix      Prefix and suffix of the print  
     */
    template <typename DataType>
    void printMatrix(std::vector<DataType> const &dataPoints,
                     int const dataPointsDim,
                     std::vector<DataType> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data with stream format for each
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints              Input data points
     * \param dataPointsDim           Number of columns of input array data (data dimension)
     * \param functionValues          Array of input value data
     * \param nFunctionValues         Number of columns of input value data (functionValues at data points)
     * \param nRows                   Number of rows
     * \param dataIOFormat            Print format for input data
     * \param functionValuesIOFormat  Print format for input function value 
     */
    template <typename DataType>
    void printMatrix(DataType const *dataPoints,
                     int const dataPointsDim,
                     DataType const *functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     ioFormat const &dataIOFormat,
                     ioFormat const &functionValuesIOFormat);

    /*!
     * \brief Helper function to print two vectors of data with stream format for each
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints              Input data points
     * \param dataPointsDim           Number of columns of input array data (data dimension)
     * \param functionValues          Array of input value data
     * \param nFunctionValues         Number of columns of input value data (functionValues at data points)
     * \param nRows                   Number of rows
     * \param dataIOFormat            Print format for input data
     * \param functionValuesIOFormat  Print format for input function value 
     */
    template <typename DataType>
    void printMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                     int const dataPointsDim,
                     std::unique_ptr<DataType[]> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     ioFormat const &dataIOFormat,
                     ioFormat const &functionValuesIOFormat);

    /*!
     * \brief Helper function to print two vectors of data with stream format for each
     * 
     * \tparam DataType Data type
     * 
     * \param dataPoints              Input data points
     * \param dataPointsDim           Number of columns of input array data (data dimension)
     * \param functionValues          Array of input value data
     * \param nFunctionValues         Number of columns of input value data (functionValues at data points)
     * \param nRows                   Number of rows
     * \param dataIOFormat            Print format for input data
     * \param functionValuesIOFormat  Print format for input function value 
     */
    template <typename DataType>
    void printMatrix(std::vector<DataType> const &dataPoints,
                     int const dataPointsDim,
                     std::vector<DataType> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     ioFormat const &dataIOFormat,
                     ioFormat const &functionValuesIOFormat);

  private:
    //! Input/output file based stream
    std::fstream fs;

    //! Line for reading the string of data
    std::string line;

    //! Index type
    typedef std::ptrdiff_t Idx;

    //! Width parameter of the stream \c out or stream \c in
    std::ptrdiff_t Width;

    //! Flag for using the pre defined stream Width or computing it on the fly from the input data
    bool FixedWidth;

    //! IO format
    ioFormat fmt;
};

ioFormat::ioFormat(std::string const &CoeffSeparator,
                   std::string const &RowSeparator,
                   std::string const &RowPrefix,
                   std::string const &RowSuffix) : coeffSeparator(CoeffSeparator),
                                                   rowSeparator(RowSeparator),
                                                   rowPrefix(RowPrefix),
                                                   rowSuffix(RowSuffix) {}

inline bool ioFormat::operator==(ioFormat const &rhs)
{
    return coeffSeparator == rhs.coeffSeparator &&
           rowSeparator == rhs.rowSeparator &&
           rowPrefix == rhs.rowPrefix &&
           rowSuffix == rhs.rowSuffix;
}

inline bool ioFormat::operator==(ioFormat const &rhs) const
{
    return coeffSeparator == rhs.coeffSeparator &&
           rowSeparator == rhs.rowSeparator &&
           rowPrefix == rhs.rowPrefix &&
           rowSuffix == rhs.rowSuffix;
}

inline bool ioFormat::operator!=(ioFormat const &rhs)
{
    return coeffSeparator != rhs.coeffSeparator ||
           rowSeparator != rhs.rowSeparator ||
           rowPrefix != rhs.rowPrefix ||
           rowSuffix != rhs.rowSuffix;
}

inline bool ioFormat::operator!=(ioFormat const &rhs) const
{
    return coeffSeparator != rhs.coeffSeparator ||
           rowSeparator != rhs.rowSeparator ||
           rowPrefix != rhs.rowPrefix ||
           rowSuffix != rhs.rowSuffix;
}

io::io() : Width(0), FixedWidth(false) {}

io::~io()
{
    io::closeFile();
}

inline bool io::isFileOpened() const { return fs.is_open(); }

inline bool io::isFileExist(const char *fileName)
{
    struct stat buffer;
    if (stat(fileName, &buffer))
    {
        UMUQFAILRETURN(std::string("The requested file [") + std::string(fileName) + std::string("] does not exist in the current PATH!!"));
    }
    return true;
}

inline bool io::isFileExist(std::string const &fileName)
{
    return io::isFileExist(&fileName[0]);
}

bool io::openFile(const char *fileName, const std::ios_base::openmode mode)
{
    if (fs.is_open())
    {
        UMUQFAILRETURN("The requested File is already open by another stream!");
    }

    fs.open(fileName, mode);
    if (!fs.is_open())
    {
        UMUQFAILRETURN("The requested File to open does not exists!");
    }

    // Returns false if an error has occurred on the associated stream.
    if (fs.fail())
    {
        UMUQFAILRETURN("An error has occurred on the associated stream from opening the file!");
    }

    return true;
}

bool io::openFile(std::string const &fileName, const std::ios_base::openmode mode)
{
    return openFile(&fileName[0], mode);
}

bool io::readLine(const char comment)
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

inline void io::rewindFile()
{
    // Clearing all error state flags if there is any
    fs.clear();

    // Rewind the file
    fs.seekg(0);
    return;
}

inline void io::closeFile()
{
    fs.close();
    return;
}

inline std::fstream &io::getFstream() { return fs; }

inline std::string &io::getLine() { return line; }

template <typename DataType>
inline void io::setPrecision(std::ostream &streamBuffer)
{
    if (std::numeric_limits<DataType>::is_integer)
    {
        // Manages the precision (i.e. how many digits are generated)
        streamBuffer.precision(0);
    }
    else
    {
        // Manages the precision (i.e. how many digits are generated)
        streamBuffer.precision(digits10<DataType>());
        streamBuffer << std::fixed;
    }
}

inline void io::setWidth(int const streamWidth)
{
    Width = streamWidth < 0 ? std::ptrdiff_t{} : static_cast<std::ptrdiff_t>(streamWidth);
    FixedWidth = streamWidth >= 0;
}

template <typename DataType>
int io::getWidth(DataType const *dataPoints, int const nRows, int const nCols, std::ostream &streamBuffer)
{
    std::ptrdiff_t tWidth(0);
    setPrecision<DataType>(streamBuffer);
    for (int i = 0; i < nRows * nCols; i++)
    {
        std::stringstream sstr;
        sstr.copyfmt(streamBuffer);
        sstr << dataPoints[i];
        tWidth = std::max<std::ptrdiff_t>(tWidth, io::Idx(sstr.str().length()));
    }
    return static_cast<int>(tWidth);
}

template <typename DataType>
int io::getWidth(std::unique_ptr<DataType[]> const &dataPoints, int const nRows, int const nCols, std::ostream &streamBuffer)
{
    std::ptrdiff_t tWidth(0);
    setPrecision<DataType>(streamBuffer);
    for (int i = 0; i < nRows * nCols; i++)
    {
        std::stringstream sstr;
        sstr.copyfmt(streamBuffer);
        sstr << dataPoints[i];
        tWidth = std::max<std::ptrdiff_t>(tWidth, io::Idx(sstr.str().length()));
    }
    return static_cast<int>(tWidth);
}

template <typename DataType>
int io::getWidth(std::vector<DataType> const &dataPoints, int const nRows, int const nCols, std::ostream &streamBuffer)
{
    std::ptrdiff_t tWidth(0);
    setPrecision<DataType>(streamBuffer);
    for (int i = 0; i < nRows * nCols; i++)
    {
        std::stringstream sstr;
        sstr.copyfmt(streamBuffer);
        sstr << dataPoints[i];
        tWidth = std::max<std::ptrdiff_t>(tWidth, io::Idx(sstr.str().length()));
    }
    return static_cast<int>(tWidth);
}

template <typename DataType>
int io::getWidth(DataType **dataPoints, int const nRows, int const nCols, std::ostream &streamBuffer)
{
    std::ptrdiff_t tWidth(0);
    setPrecision<DataType>(streamBuffer);
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            std::stringstream sstr;
            sstr.copyfmt(streamBuffer);
            sstr << dataPoints[i][j];
            tWidth = std::max<std::ptrdiff_t>(tWidth, io::Idx(sstr.str().length()));
        }
    }
    return static_cast<int>(tWidth);
}

template <typename EigenMatrixType, typename IOFormatType>
bool io::saveMatrix(EigenMatrixType &dataMatrix, IOFormatType const &ioformat)
{
    if (fs.is_open())
    {
        fs << std::fixed;
        fs << dataMatrix.format(ioformat);
        fs << fmt.rowSeparator;

        return true;
    }
    UMUQFAILRETURN("This file stream is not open for writing!");
}

template <typename DataType>
bool io::saveMatrix(DataType **dataPoints, int const nRows, int const nCols, int const options)
{
    if (!fs.is_open())
    {
        UMUQFAILRETURN("This file stream is not open for writing!");
    }

    std::string rowSeparator;
    if (options > 0)
    {
        rowSeparator = fmt.rowSeparator;
        fmt.rowSeparator = fmt.coeffSeparator;
    }

    setPrecision<DataType>(fs);

    if (!FixedWidth)
    {
        Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(fs);
                sstr << dataPoints[i][j];
                Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
            }
        }
    }

    if (Width)
    {
        for (int i = 0; i < nRows; ++i)
        {
            fs << fmt.rowPrefix;
            fs.width(Width);
            fs << dataPoints[i][0];
            for (int j = 1; j < nCols; ++j)
            {
                fs << fmt.coeffSeparator;
                fs.width(Width);
                fs << dataPoints[i][j];
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
            fs << dataPoints[i][0];
            for (int j = 1; j < nCols; ++j)
            {
                fs << fmt.coeffSeparator;
                fs << dataPoints[i][j];
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

template <typename DataType>
bool io::saveMatrix(DataType **dataPoints,
                    int const nRows,
                    int const *nCols,
                    int const options,
                    int const entries,
                    std::vector<ioFormat> const &ioformat)
{
    if (!fs.is_open())
    {
        UMUQFAILRETURN("This file stream is not open for writing!");
    }

    setPrecision<DataType>(fs);

    //Default case, only one set of data
    if (entries == 1)
    {
        if (ioformat.size() != static_cast<decltype(ioformat.size())>(nRows))
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
                        sstr << dataPoints[i][j];
                        Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
                    }
                }
            }

            if (Width)
            {
                for (int i = 0; i < nRows; ++i)
                {
                    fs << fmt.rowPrefix;
                    fs.width(Width);
                    fs << dataPoints[i][0];
                    for (int j = 1; j < nCols[i]; ++j)
                    {
                        fs << fmt.coeffSeparator;
                        fs.width(Width);
                        fs << dataPoints[i][j];
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
                    fs << dataPoints[i][0];
                    for (int j = 1; j < nCols[i]; ++j)
                    {
                        fs << fmt.coeffSeparator;
                        fs << dataPoints[i][j];
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
                        sstr << dataPoints[i][j];
                        Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
                    }
                }
            }

            if (Width)
            {
                for (int i = 0; i < nRows; ++i)
                {
                    fs << ioformat[i].rowPrefix;
                    fs.width(Width);
                    fs << dataPoints[i][0];
                    for (int j = 1; j < nCols[i]; ++j)
                    {
                        fs << ioformat[i].coeffSeparator;
                        fs.width(Width);
                        fs << dataPoints[i][j];
                    }
                    fs << ioformat[i].rowSuffix;
                    fs << ioformat[i].rowSeparator;
                }
            }
            else
            {
                for (int i = 0; i < nRows; ++i)
                {
                    fs << ioformat[i].rowPrefix;
                    fs << dataPoints[i][0];
                    for (int j = 1; j < nCols[i]; ++j)
                    {
                        fs << ioformat[i].coeffSeparator;
                        fs << dataPoints[i][j];
                    }
                    fs << ioformat[i].rowSuffix;
                    fs << ioformat[i].rowSeparator;
                }
            }

            return true;
        }
    }
    else
    {
        if (ioformat.size() != static_cast<decltype(ioformat.size())>(nRows))
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
                    DataType *ePointer = dataPoints[i];
                    for (int e = 0; e < entries; e++)
                    {
                        for (int j = 0; j < nCols[i]; j++)
                        {
                            std::stringstream sstr;
                            sstr.copyfmt(fs);
                            sstr << *ePointer++;
                            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
                        }
                    }
                }
            }

            if (Width)
            {
                if (options == 1)
                {
                    DataType *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = dataPoints[i] + e * nCols[i];

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
                    DataType *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = dataPoints[i] + e * nCols[i];

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
                    }
                }
            }
            else
            {
                if (options == 1)
                {
                    DataType *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = dataPoints[i] + e * nCols[i];

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
                else
                {
                    DataType *ePointer;
                    for (int e = 0; e < entries; e++)
                    {
                        for (int i = 0; i < nRows; ++i)
                        {
                            ePointer = dataPoints[i] + e * nCols[i];

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
                    DataType *ePointer = dataPoints[i];
                    for (int e = 0; e < entries; e++)
                    {
                        for (int j = 0; j < nCols[i]; j++)
                        {
                            std::stringstream sstr;
                            sstr.copyfmt(fs);
                            sstr << *ePointer++;
                            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
                        }
                    }
                }
            }

            if (Width)
            {
                DataType *ePointer;
                for (int e = 0; e < entries; e++)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        ePointer = dataPoints[i] + e * nCols[i];

                        fs << ioformat[i].rowPrefix;
                        fs.width(Width);
                        fs << *ePointer++;
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            fs << ioformat[i].coeffSeparator;
                            fs.width(Width);
                            fs << *ePointer++;
                        }
                        fs << ioformat[i].rowSuffix;
                        fs << ioformat[i].rowSeparator;
                    }
                }
            }
            else
            {
                DataType *ePointer;
                for (int e = 0; e < entries; e++)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        ePointer = dataPoints[i] + e * nCols[i];

                        fs << ioformat[i].rowPrefix;
                        fs << *ePointer++;
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            fs << ioformat[i].coeffSeparator;
                            fs << *ePointer++;
                        }
                        fs << ioformat[i].rowSuffix;
                        fs << ioformat[i].rowSeparator;
                    }
                }
            }

            return true;
        }
    }
}

template <typename DataType>
bool io::saveMatrix(DataType const *dataPoints,
                    int const nRows,
                    int const nCols,
                    int const options)
{
    if (!fs.is_open())
    {
        UMUQFAILRETURN("This file stream is not open for writing!");
    }

    std::string rowSeparator;
    if (options > 0)
    {
        rowSeparator = fmt.rowSeparator;
        fmt.rowSeparator = fmt.coeffSeparator;
    }

    setPrecision<DataType>(fs);

    if (!FixedWidth)
    {
        Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

        for (int i = 0; i < nRows * nCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << dataPoints[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }
    }

    if (Width)
    {
        if (nCols == 1)
        {
            fs << fmt.rowPrefix;
            fs.width(Width);
            fs << dataPoints[0];
            for (int i = 1; i < nRows; i++)
            {
                fs << fmt.coeffSeparator;
                fs.width(Width);
                fs << dataPoints[i];
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
                fs << dataPoints[l++];
                for (int j = 1; j < nCols; j++)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << dataPoints[l++];
                }
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
            fs << dataPoints[0];
            for (int i = 1; i < nRows; i++)
            {
                fs << fmt.coeffSeparator;
                fs << dataPoints[i];
            }
            fs << fmt.rowSuffix;
            fs << fmt.rowSeparator;
        }
        else
        {
            for (int i = 0, l = 0; i < nRows; i++)
            {
                fs << fmt.rowPrefix;
                fs << dataPoints[l++];
                for (int j = 1; j < nCols; j++)
                {
                    fs << fmt.coeffSeparator;
                    fs << dataPoints[l++];
                }
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

template <typename DataType>
bool io::saveMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                    int const nRows,
                    int const nCols,
                    int const options)
{
    return io::saveMatrix<DataType>(dataPoints.get(), nRows, nCols, options);
}

template <typename DataType>
bool io::saveMatrix(std::vector<DataType> const &dataPoints,
                    int const nRows,
                    int const nCols,
                    int const options)
{
    return io::saveMatrix<DataType>(const_cast<DataType *>(dataPoints.data()), nRows, nCols, options);
}

template <typename DataType>
bool io::saveMatrix(DataType const *dataPoints,
                    int const dataPointsDim,
                    DataType const *functionValues,
                    int const nFunctionValues,
                    int const nRows)
{
    if (!fs.is_open())
    {
        UMUQFAILRETURN("This file stream is not open for writing!");
    }

    setPrecision<DataType>(fs);

    if (!FixedWidth)
    {
        Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

        for (int i = 0; i < nRows * dataPointsDim; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << dataPoints[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }

        for (int i = 0; i < nRows * nFunctionValues; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << functionValues[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }
    }

    if (Width)
    {
        for (int i = 0, l = 0, k = 0; i < nRows; i++)
        {
            fs << fmt.rowPrefix;
            fs.width(Width);
            fs << dataPoints[l++];
            for (int j = 1; j < dataPointsDim; j++)
            {
                fs << fmt.coeffSeparator;
                fs.width(Width);
                fs << dataPoints[l++];
            }
            for (int j = 0; j < nFunctionValues; j++)
            {
                fs << fmt.coeffSeparator;
                fs.width(Width);
                fs << functionValues[k++];
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
            fs << dataPoints[l++];
            for (int j = 1; j < dataPointsDim; j++)
            {
                fs << fmt.coeffSeparator;
                fs << dataPoints[l++];
            }
            for (int j = 0; j < nFunctionValues; j++)
            {
                fs << fmt.coeffSeparator;
                fs << functionValues[k++];
            }
            fs << fmt.rowSuffix;
            fs << fmt.rowSeparator;
        }
    }

    return true;
}

template <typename DataType>
bool io::saveMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                    int const dataPointsDim,
                    std::unique_ptr<DataType[]> const &functionValues,
                    int const nFunctionValues,
                    int const nRows)
{
    return io::saveMatrix<DataType>(dataPoints.get(), dataPointsDim, functionValues.get(), nFunctionValues, nRows);
}

template <typename DataType>
bool io::saveMatrix(std::vector<DataType> const &dataPoints,
                    int const dataPointsDim,
                    std::vector<DataType> const &functionValues,
                    int const nFunctionValues,
                    int const nRows)
{
    return io::saveMatrix<DataType>(dataPoints.data(), dataPointsDim, functionValues.data(), nFunctionValues, nRows);
}

template <typename EigenMatrixType>
bool io::loadMatrix(EigenMatrixType &dataMatrix)
{
    std::string Line;

    for (int i = 0; i < dataMatrix.rows(); i++)
    {
        if (std::getline(fs, Line))
        {
            std::stringstream inLine(Line);

            for (int j = 0; j < dataMatrix.cols(); j++)
            {
                inLine >> dataMatrix(i, j);
            }
        }
        else
        {
            return false;
        }
    }
    return true;
}

template <typename DataType>
bool io::loadMatrix(DataType **dataPoints, int const nRows, int const nCols, int const options)
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
                    inLine >> dataPoints[i][j];
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
                    inLine >> dataPoints[i][j];
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

template <typename DataType>
bool io::loadMatrix(DataType **dataPoints, int const nRows, int const *nCols, int const options, int const entries)
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
                        inLine >> dataPoints[i][j];
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
                        inLine >> dataPoints[i][j];
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
            DataType *ePointer;
            for (int e = 0; e < entries; e++)
            {
                for (int i = 0; i < nRows; i++)
                {
                    if (std::getline(fs, Line))
                    {
                        std::stringstream inLine(Line);

                        ePointer = dataPoints[i] + e * nCols[i];

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
            DataType *ePointer;
            for (int e = 0; e < entries; e++)
            {
                if (std::getline(fs, Line))
                {
                    std::stringstream inLine(Line);

                    for (int i = 0; i < nRows; i++)
                    {
                        ePointer = dataPoints[i] + e * nCols[i];

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
            }
            return true;
        }
        return false;
    }
}

template <typename DataType>
bool io::loadMatrix(DataType *dataPoints, int const nRows, int const nCols)
{
    std::string Line;

    if (nCols == 1)
    {
        if (std::getline(fs, Line))
        {
            std::stringstream inLine(Line);
            for (int i = 0; i < nRows; i++)
            {
                inLine >> dataPoints[i];
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

                for (int j = 0; j < nCols; j++)
                {
                    inLine >> dataPoints[l++];
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

template <typename DataType>
bool io::loadMatrix(std::unique_ptr<DataType[]> &dataPoints, int const nRows, int const nCols)
{
    return io::loadMatrix<DataType>(dataPoints.get(), nRows, nCols);
}

template <typename DataType>
bool io::loadMatrix(std::vector<DataType> &dataPoints, int const nRows, int const nCols)
{
    return io::loadMatrix<DataType>(dataPoints.data(), nRows, nCols);
}

template <typename DataType>
bool io::loadMatrix(DataType *dataPoints, int const dataPointsDim, DataType *functionValues, int const nFunctionValues, int const nRows)
{
    std::string Line;

    for (int i = 0, k = 0, l = 0; i < nRows; i++)
    {
        if (std::getline(fs, Line))
        {
            std::stringstream inLine(Line);

            for (int j = 0; j < dataPointsDim; j++)
            {
                inLine >> dataPoints[k++];
            }

            for (int j = 0; j < nFunctionValues; j++)
            {
                inLine >> functionValues[l++];
            }
        }
        else
        {
            return false;
        }
    }
    return true;
}

template <typename DataType>
bool io::loadMatrix(std::unique_ptr<DataType[]> &dataPoints,
                    int const dataPointsDim,
                    std::unique_ptr<DataType[]> &functionValues,
                    int const nFunctionValues,
                    int const nRows)
{
    return io::loadMatrix<DataType>(dataPoints.get(), dataPointsDim, functionValues.get(), nFunctionValues, nRows);
}

template <typename DataType>
bool io::loadMatrix(std::vector<DataType> &dataPoints,
                    int const dataPointsDim,
                    std::vector<DataType> &functionValues,
                    int const nFunctionValues,
                    int const nRows)
{
    return io::loadMatrix<DataType>(dataPoints.data(), dataPointsDim, functionValues.data(), nFunctionValues, nRows);
}

template <typename DataType>
void io::printMatrix(const char *title,
                     DataType **dataPoints,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix)
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

    setPrecision<DataType>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(std::cout);
                sstr << dataPoints[i][j];
                Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
            }
        }
    }

    if (Width)
    {
        for (int i = 0; i < nRows; ++i)
        {
            std::cout << fmt.rowPrefix;
            std::cout.width(Width);
            std::cout << dataPoints[i][0];
            for (int j = 1; j < nCols; ++j)
            {
                std::cout << fmt.coeffSeparator;
                std::cout.width(Width);
                std::cout << dataPoints[i][j];
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
            std::cout << dataPoints[i][0];
            for (int j = 1; j < nCols; ++j)
            {
                std::cout << fmt.coeffSeparator;
                std::cout << dataPoints[i][j];
            }
            std::cout << fmt.rowSuffix;
            std::cout << fmt.rowSeparator;
        }
    }
    std::cout << printPrefix;
}

template <typename DataType>
void io::printMatrix(DataType **dataPoints,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>("", dataPoints, nRows, nCols, printPrefix);
}

template <typename DataType>
void io::printMatrix(DataType **dataPoints,
                     int const nRows,
                     int const nCols,
                     ioFormat const &ioformat)
{
    setPrecision<DataType>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(std::cout);
                sstr << dataPoints[i][j];
                Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
            }
        }
    }

    if (Width)
    {
        for (int i = 0; i < nRows; ++i)
        {
            std::cout << ioformat.rowPrefix;
            std::cout.width(Width);
            std::cout << dataPoints[i][0];
            for (int j = 1; j < nCols; ++j)
            {
                std::cout << ioformat.coeffSeparator;
                std::cout.width(Width);
                std::cout << dataPoints[i][j];
            }
            std::cout << ioformat.rowSuffix;
            std::cout << ioformat.rowSeparator;
        }
    }
    else
    {
        for (int i = 0; i < nRows; ++i)
        {
            std::cout << ioformat.rowPrefix;
            std::cout << dataPoints[i][0];
            for (int j = 1; j < nCols; ++j)
            {
                std::cout << ioformat.coeffSeparator;
                std::cout << dataPoints[i][j];
            }
            std::cout << ioformat.rowSuffix;
            std::cout << ioformat.rowSeparator;
        }
    }
}

template <typename DataType>
void io::printMatrix(const char *title,
                     DataType **dataPoints,
                     int const nRows,
                     int const *nCols,
                     int const entries,
                     std::string const &printPrefix)
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

    setPrecision<DataType>(std::cout);

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
                    sstr << dataPoints[i][j];
                    Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
                }
            }
        }

        if (Width)
        {
            for (int i = 0; i < nRows; ++i)
            {
                std::cout << fmt.rowPrefix;
                std::cout.width(Width);
                std::cout << dataPoints[i][0];
                for (int j = 1; j < nCols[i]; ++j)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << dataPoints[i][j];
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
                std::cout << dataPoints[i][0];
                for (int j = 1; j < nCols[i]; ++j)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout << dataPoints[i][j];
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
                DataType *ePointer = dataPoints[i];
                for (int e = 0; e < entries; e++)
                {
                    for (int j = 0; j < nCols[i]; j++)
                    {
                        std::stringstream sstr;
                        sstr.copyfmt(std::cout);
                        sstr << *ePointer++;
                        Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
                    }
                }
            }
        }

        if (Width)
        {
            DataType *ePointer;
            for (int e = 0; e < entries; e++)
            {
                for (int i = 0; i < nRows; ++i)
                {
                    ePointer = dataPoints[i] + e * nCols[i];

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
            DataType *ePointer;
            for (int e = 0; e < entries; e++)
            {
                for (int i = 0; i < nRows; ++i)
                {
                    ePointer = dataPoints[i] + e * nCols[i];

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

template <typename DataType>
void io::printMatrix(DataType **dataPoints,
                     int const nRows,
                     int const *nCols,
                     int const entries,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>("", dataPoints, nRows, nCols, entries, printPrefix);
}

template <typename DataType>
void io::printMatrix(DataType **dataPoints,
                     int const nRows,
                     int const *nCols,
                     int const entries,
                     std::vector<ioFormat> const &ioformat)
{
    if (ioformat.size() != static_cast<decltype(ioformat.size())>(nRows))
    {
        io::printMatrix<DataType>(dataPoints, nRows, nCols, entries);
    }
    else
    {
        setPrecision<DataType>(std::cout);

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
                        sstr << dataPoints[i][j];
                        Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
                    }
                }
            }

            if (Width)
            {
                for (int i = 0; i < nRows; ++i)
                {
                    std::cout << ioformat[i].rowPrefix;
                    std::cout.width(Width);
                    std::cout << dataPoints[i][0];
                    for (int j = 1; j < nCols[i]; ++j)
                    {
                        std::cout << ioformat[i].coeffSeparator;
                        std::cout.width(Width);
                        std::cout << dataPoints[i][j];
                    }
                    std::cout << ioformat[i].rowSuffix;
                    std::cout << ioformat[i].rowSeparator;
                }
            }
            else
            {
                for (int i = 0; i < nRows; ++i)
                {
                    std::cout << ioformat[i].rowPrefix;
                    std::cout << dataPoints[i][0];
                    for (int j = 1; j < nCols[i]; ++j)
                    {
                        std::cout << ioformat[i].coeffSeparator;
                        std::cout << dataPoints[i][j];
                    }
                    std::cout << ioformat[i].rowSuffix;
                    std::cout << ioformat[i].rowSeparator;
                }
            }
        }
        else
        {
            if (!FixedWidth)
            {
                for (int i = 0; i < nRows; i++)
                {
                    DataType *ePointer = dataPoints[i];
                    for (int e = 0; e < entries; e++)
                    {
                        for (int j = 0; j < nCols[i]; j++)
                        {
                            std::stringstream sstr;
                            sstr.copyfmt(std::cout);
                            sstr << *ePointer++;
                            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
                        }
                    }
                }
            }

            if (Width)
            {
                DataType *ePointer;
                for (int e = 0; e < entries; e++)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        ePointer = dataPoints[i] + e * nCols[i];

                        std::cout << ioformat[i].rowPrefix;
                        std::cout.width(Width);
                        std::cout << *ePointer++;
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            std::cout << ioformat[i].coeffSeparator;
                            std::cout.width(Width);
                            std::cout << *ePointer++;
                        }
                        std::cout << ioformat[i].rowSuffix;
                        std::cout << ioformat[i].rowSeparator;
                    }
                }
            }
            else
            {
                DataType *ePointer;
                for (int e = 0; e < entries; e++)
                {
                    for (int i = 0; i < nRows; ++i)
                    {
                        ePointer = dataPoints[i] + e * nCols[i];

                        std::cout << ioformat[i].rowPrefix;
                        std::cout << *ePointer++;
                        for (int j = 1; j < nCols[i]; ++j)
                        {
                            std::cout << ioformat[i].coeffSeparator;
                            std::cout << *ePointer++;
                        }
                        std::cout << ioformat[i].rowSuffix;
                        std::cout << ioformat[i].rowSeparator;
                    }
                }
            }
        }
    }
}

template <typename DataType>
void io::printMatrix(const char *title,
                     DataType const *dataPoints,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix)
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

    setPrecision<DataType>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows * nCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << dataPoints[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }
    }

    if (nCols == 1)
    {
        std::cout << fmt.rowPrefix;
        if (Width)
        {
            std::cout.width(Width);
            std::cout << dataPoints[0];
            for (int i = 1; i < nRows; i++)
            {
                std::cout << fmt.coeffSeparator;
                std::cout.width(Width);
                std::cout << dataPoints[i];
            }
        }
        else
        {
            std::cout << dataPoints[0];
            for (int i = 1; i < nRows; i++)
            {
                std::cout << fmt.coeffSeparator;
                std::cout << dataPoints[i];
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
                std::cout << dataPoints[l++];
                for (int j = 1; j < nCols; j++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << dataPoints[l++];
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
                std::cout << dataPoints[l++];
                for (int j = 1; j < nCols; j++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout << dataPoints[l++];
                }
                std::cout << fmt.rowSuffix;
                std::cout << fmt.rowSeparator;
            }
        }
    }
    std::cout << printPrefix;
}

template <typename DataType>
void io::printMatrix(const char *title,
                     std::unique_ptr<DataType[]> const &dataPoints,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>(title, dataPoints.get(), nRows, nCols, printPrefix);
}

template <typename DataType>
void io::printMatrix(const char *title,
                     std::vector<DataType> const &dataPoints,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>(title, dataPoints.data(), nRows, nCols, printPrefix);
}

template <typename DataType>
void io::printMatrix(DataType const *dataPoints,
                     int const nRows,
                     int const nCols,
                     ioFormat const &ioformat)
{
    if (ioformat.coeffSeparator == "NO")
    {
        io::printMatrix<DataType>("", dataPoints, nRows, nCols);
    }
    else
    {
        setPrecision<DataType>(std::cout);

        if (!FixedWidth)
        {
            for (int i = 0; i < nRows * nCols; i++)
            {
                std::stringstream sstr;
                sstr.copyfmt(std::cout);
                sstr << dataPoints[i];
                Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
            }
        }

        if (nCols == 1)
        {
            std::cout << ioformat.rowPrefix;
            if (Width)
            {
                std::cout.width(Width);
                std::cout << dataPoints[0];
                for (int i = 1; i < nRows; i++)
                {
                    std::cout << ioformat.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << dataPoints[i];
                }
            }
            else
            {
                std::cout << dataPoints[0];
                for (int i = 1; i < nRows; i++)
                {
                    std::cout << ioformat.coeffSeparator;
                    std::cout << dataPoints[i];
                }
            }
            std::cout << ioformat.rowSuffix;
            std::cout << ioformat.rowSeparator;
        }
        else
        {
            if (Width)
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    std::cout << ioformat.rowPrefix;
                    std::cout.width(Width);
                    std::cout << dataPoints[l++];
                    for (int j = 1; j < nCols; j++)
                    {
                        std::cout << ioformat.coeffSeparator;
                        std::cout.width(Width);
                        std::cout << dataPoints[l++];
                    }
                    std::cout << ioformat.rowSuffix;
                    std::cout << ioformat.rowSeparator;
                }
            }
            else
            {
                for (int i = 0, l = 0; i < nRows; i++)
                {
                    std::cout << ioformat.rowPrefix;
                    std::cout << dataPoints[l++];
                    for (int j = 1; j < nCols; j++)
                    {
                        std::cout << ioformat.coeffSeparator;
                        std::cout << dataPoints[l++];
                    }
                    std::cout << ioformat.rowSuffix;
                    std::cout << ioformat.rowSeparator;
                }
            }
        }
    }
}

template <typename DataType>
void io::printMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                     int const nRows,
                     int const nCols,
                     ioFormat const &ioformat)
{
    io::printMatrix<DataType>(dataPoints.get(), nRows, nCols, ioformat);
}

template <typename DataType>
void io::printMatrix(std::vector<DataType> const &dataPoints,
                     int const nRows,
                     int const nCols,
                     ioFormat const &ioformat)
{
    io::printMatrix<DataType>(dataPoints.data(), nRows, nCols, ioformat);
}

template <typename DataType>
void io::printMatrix(DataType const *dataPoints, ioFormat const &ioformat)
{
    setPrecision<DataType>(std::cout);

    if (!FixedWidth)
    {
        std::stringstream sstr;
        sstr.copyfmt(std::cout);
        sstr << *dataPoints;
        Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
    }

    std::cout << ioformat.rowPrefix;
    if (Width)
    {
        std::cout.width(Width);
    }
    std::cout << *dataPoints;
    std::cout << ioformat.rowSuffix;
    std::cout << ioformat.rowSeparator;
}

template <typename DataType>
void io::printMatrix(const char *title,
                     DataType const *dataPoints,
                     int const dataPointsDim,
                     DataType const *functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix)
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

    setPrecision<DataType>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows * dataPointsDim; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << dataPoints[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }

        for (int i = 0; i < nRows * nFunctionValues; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << functionValues[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }
    }

    if (Width)
    {
        for (int i = 0, l = 0, k = 0; i < nRows; i++)
        {
            std::cout << fmt.rowPrefix;
            std::cout.width(Width);
            std::cout << dataPoints[l++];
            for (int j = 1; j < dataPointsDim; j++)
            {
                std::cout << fmt.coeffSeparator;
                std::cout.width(Width);
                std::cout << dataPoints[l++];
            }
            for (int j = 0; j < nFunctionValues; j++)
            {
                std::cout << fmt.coeffSeparator;
                std::cout.width(Width);
                std::cout << functionValues[k++];
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
            std::cout << dataPoints[l++];
            for (int j = 1; j < dataPointsDim; j++)
            {
                std::cout << fmt.coeffSeparator;
                std::cout << dataPoints[l++];
            }
            for (int j = 0; j < nFunctionValues; j++)
            {
                std::cout << fmt.coeffSeparator;
                std::cout << functionValues[k++];
            }
            std::cout << fmt.rowSuffix;
            std::cout << fmt.rowSeparator;
        }
    }
    std::cout << printPrefix;
}

template <typename DataType>
void io::printMatrix(const char *title,
                     std::unique_ptr<DataType[]> const &dataPoints,
                     int const dataPointsDim,
                     std::unique_ptr<DataType[]> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>(title, dataPoints.get(), dataPointsDim, functionValues.get(), nFunctionValues, nRows, printPrefix);
}

template <typename DataType>
void io::printMatrix(const char *title,
                     std::vector<DataType> const &dataPoints,
                     int const dataPointsDim,
                     std::vector<DataType> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>(title, dataPoints.data(), dataPointsDim, functionValues.data(), nFunctionValues, nRows, printPrefix);
}

template <typename DataType>
void io::printMatrix(DataType const *dataPoints,
                     int const dataPointsDim, DataType *functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>("", dataPoints, dataPointsDim, functionValues, nFunctionValues, nRows, printPrefix);
}

template <typename DataType>
void io::printMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                     int const dataPointsDim,
                     std::unique_ptr<DataType[]> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>("", dataPoints.get(), dataPointsDim, functionValues.get(), nFunctionValues, nRows, printPrefix);
}

template <typename DataType>
void io::printMatrix(std::vector<DataType> const &dataPoints,
                     int const dataPointsDim,
                     std::vector<DataType> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<DataType>("", dataPoints.data(), dataPointsDim, functionValues.data(), nFunctionValues, nRows, printPrefix);
}

template <typename DataType>
void io::printMatrix(DataType const *dataPoints,
                     int const dataPointsDim,
                     DataType const *functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     ioFormat const &dataIOFormat,
                     ioFormat const &functionValuesIOFormat)
{
    setPrecision<DataType>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows * dataPointsDim; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << dataPoints[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }

        for (int i = 0; i < nRows * nFunctionValues; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << functionValues[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }
    }

    if (Width)
    {
        for (int i = 0, l = 0, k = 0; i < nRows; i++)
        {
            std::cout << dataIOFormat.rowPrefix;
            std::cout.width(Width);
            std::cout << dataPoints[l++];
            for (int j = 1; j < dataPointsDim; j++)
            {
                std::cout << dataIOFormat.coeffSeparator;
                std::cout.width(Width);
                std::cout << dataPoints[l++];
            }
            std::cout << dataIOFormat.rowSuffix;
            std::cout << dataIOFormat.rowSeparator;
            std::cout << functionValuesIOFormat.rowPrefix;
            for (int j = 0; j < nFunctionValues; j++)
            {
                std::cout << functionValuesIOFormat.coeffSeparator;
                std::cout.width(Width);
                std::cout << functionValues[k++];
            }
            std::cout << functionValuesIOFormat.rowSuffix;
            std::cout << functionValuesIOFormat.rowSeparator;
        }
    }
    else
    {
        for (int i = 0, l = 0, k = 0; i < nRows; i++)
        {
            std::cout << dataIOFormat.rowPrefix;
            std::cout << dataPoints[l++];
            for (int j = 1; j < dataPointsDim; j++)
            {
                std::cout << dataIOFormat.coeffSeparator;
                std::cout << dataPoints[l++];
            }
            std::cout << dataIOFormat.rowSuffix;
            std::cout << dataIOFormat.rowSeparator;
            std::cout << functionValuesIOFormat.rowPrefix;
            for (int j = 0; j < nFunctionValues; j++)
            {
                std::cout << functionValuesIOFormat.coeffSeparator;
                std::cout << functionValues[k++];
            }
            std::cout << functionValuesIOFormat.rowSuffix;
            std::cout << functionValuesIOFormat.rowSeparator;
        }
    }
}

template <typename DataType>
void io::printMatrix(std::unique_ptr<DataType[]> const &dataPoints,
                     int const dataPointsDim,
                     std::unique_ptr<DataType[]> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     ioFormat const &dataIOFormat,
                     ioFormat const &functionValuesIOFormat)
{
    io::printMatrix<DataType>(dataPoints.get(), dataPointsDim, functionValues.get(), nFunctionValues, nRows, dataIOFormat, functionValuesIOFormat);
}

template <typename DataType>
void io::printMatrix(std::vector<DataType> const &dataPoints,
                     int const dataPointsDim,
                     std::vector<DataType> const &functionValues,
                     int const nFunctionValues,
                     int const nRows,
                     ioFormat const &dataIOFormat,
                     ioFormat const &functionValuesIOFormat)
{
    io::printMatrix<DataType>(dataPoints.data(), dataPointsDim, functionValues.data(), nFunctionValues, nRows, dataIOFormat, functionValuesIOFormat);
}

} // namespace umuq

#endif // UMUQ_IO
