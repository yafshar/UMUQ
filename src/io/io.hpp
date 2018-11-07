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
     * \returns false  
     */
    inline bool operator==(ioFormat const &rhs);
    inline bool operator==(ioFormat const &rhs) const;

    /*!
     * \brief Operator \c != compares the underlying ioFormat object
     * 
     * \param rhs  ioFormat object
     * 
     * \returns true  If lhs \c != rhs 
     * \returns false 
     */
    inline bool operator!=(ioFormat const &rhs);
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
     * \returns true if the file is already opened 
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
     * \tparam T Data type
     * 
     * \param streamBuffer  Stream buffer to use as output sequence
     */
    template <typename T>
    inline void setPrecision(std::ostream &streamBuffer);

    /*!
     * \brief Set the width parameter of the stream to exactly n.
     * 
     * If Input streamWidth is \c < 0, the function will set the stream to zero and its setting flag to false 
     * 
     * \param streamWidth New value for Width 
     */
    inline void setWidth(int streamWidth = 0);

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam T Data type
     * 
     * \param idata         Input array of data
     * \param nRows         Number of Rows
     * \param nCols         Number of Columns
     * \param streamBuffer  Stream buffer to use as output sequence 
     * 
     * \returns the width
     */
    template <typename T>
    int getWidth(T *idata, int const nRows, int const nCols, std::ostream &streamBuffer);

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam T Data type
     * 
     * \param idata         Input array of data
     * \param nRows         Number of Rows
     * \param nCols         Number of Columns
     * \param streamBuffer  Stream buffer to use as output sequence 
     * 
     * \returns the width
     */
    template <typename T>
    int getWidth(std::unique_ptr<T[]> const &idata, int const nRows, int const nCols, std::ostream &streamBuffer);

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam T Data type
     * 
     * \param idata         Input array of data
     * \param nRows         Number of Rows
     * \param nCols         Number of Columns
     * \param streamBuffer  Stream buffer to use as output sequence 
     * 
     * \returns the width
     */
    template <typename T>
    int getWidth(std::vector<T> const &idata, int const nRows, int const nCols, std::ostream &streamBuffer);

    /*!
     * \brief Get the width parameter of the input data and stream for the precision 
     * 
     * \tparam T  data type
     * 
     * \param idata         Input array of data
     * \param nRows         Number of Rows
     * \param nCols         Number of Columns
     * \param streamBuffer  Stream buffer to use as output sequence 
     * 
     * \returns The width
     */
    template <typename T>
    int getWidth(T **idata, int const nRows, int const nCols, std::ostream &streamBuffer);

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
    bool saveMatrix(TM &MX, TF const &IOfmt);

    /*!
     * \brief Helper function to save one matrix into a file 
     * 
     * \tparam T Data type 
     * 
     * \param   idata  Array of input data of type T
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns
     * \param options  (0 Default) 
     *                  - \b 0 Save matrix in matrix format and proceed the position indicator to the next line & 
     *                  - \b 1 Save matrix in vector format and proceed the position indicator to the next line &
     *                  - \b 2 Save matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true if no error occurs during writing the matrix
     */
    template <typename T>
    bool saveMatrix(T **idata, int const nRows, int const nCols, int const options = 0);

    /*!
     * \brief Helper function to save one matrix (or entries number matrices) into a file 
     * 
     * \tparam  T     Data type 
     * 
     * \param   idata   Array of input data of type T
     * \param   nRows   Number of rows
     * \param   nCols   Number of columns for each row
     * \param options   (0 Default) 
     *                      - \b 0 Saves matrix in matrix format and proceeds the position indicator to the next line &  
     *                      - \b 1 Saves matrix in vector format and proceeds the position indicator to the next line & 
     *                      - \b 2 Saves matrix in vector format and keep the position indicator on the same line
     * \param  entries  Number of data entry (Input data contains pointer to array of data)
     * \param  form     Print format for each row 
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename T>
    bool saveMatrix(T **idata,
                    int const nRows,
                    int const *nCols,
                    int const options = 0,
                    int const entries = 1,
                    std::vector<ioFormat> const &form = std::vector<ioFormat>());

    /*!
     * \brief Helper function to save the matrix or a vector into a file 
     * 
     * \tparam T      Data type 
     * 
     * \param idata    Array of input data of type T
     * \param nRows    Number of rows 
     * \param nCols    Number of columns for each row (default is 1)
     * \param options  (0 Default) 
     *                      - \b  0 Saves matrix in matrix format and proceeds the position indicator to the next line &  
     *                      - \b 1 Saves matrix in vector format and proceeds the position indicator to the next line & 
     *                      - \b 2 Saves matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename T>
    bool saveMatrix(T *idata,
                    int const nRows,
                    int const nCols = 1,
                    int const options = 0);

    /*!
     * \brief Helper function to save the matrix or a vector into a file 
     * 
     * \tparam T      Data type 
     * 
     * \param idata    Array of input data of type T
     * \param nRows    Number of rows 
     * \param nCols    Number of columns for each row (default is 1)
     * \param options  (0 Default) 
     *                      - \b  0 Saves matrix in matrix format and proceeds the position indicator to the next line &  
     *                      - \b 1 Saves matrix in vector format and proceeds the position indicator to the next line & 
     *                      - \b 2 Saves matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename T>
    bool saveMatrix(std::unique_ptr<T[]> const &idata,
                    int const nRows,
                    int const nCols = 1,
                    int const options = 0);

    /*!
     * \brief Helper function to save the matrix or a vector into a file 
     * 
     * \tparam T      Data type 
     * 
     * \param idata    Array of input data of type T
     * \param nRows    Number of rows 
     * \param nCols    Number of columns for each row (default is 1)
     * \param options  (0 Default) 
     *                      - \b  0 Saves matrix in matrix format and proceeds the position indicator to the next line &  
     *                      - \b 1 Saves matrix in vector format and proceeds the position indicator to the next line & 
     *                      - \b 2 Saves matrix in vector format and keep the position indicator on the same line
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename T>
    bool saveMatrix(std::vector<T> const &idata,
                    int const nRows,
                    int const nCols = 1,
                    int const options = 0);

    /*!
     * \brief Helper function to save two arrays of data into a file 
     * 
     * \tparam T Data type 
     * 
     * \param idata        Array of input data of type T
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type T
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows in each data set
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename T>
    bool saveMatrix(T *idata,
                    int const idataCols,
                    T *ifvalue,
                    int const ifvalueCols,
                    int const nRows);

    /*!
     * \brief Helper function to save two arrays of data into a file 
     * 
     * \tparam T Data type 
     * 
     * \param idata        Array of input data of type T
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type T
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows in each data set
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename T>
    bool saveMatrix(std::unique_ptr<T[]> const &idata,
                    int const idataCols,
                    std::unique_ptr<T[]> const &ifvalue,
                    int const ifvalueCols,
                    int const nRows);

    /*!
     * \brief Helper function to save two arrays of data into a file 
     * 
     * \tparam T Data type 
     * 
     * \param idata        Array of input data of type T
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type T
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows in each data set
     * 
     * \returns true If no error occurs during writing the matrix
     */
    template <typename T>
    bool saveMatrix(std::vector<T> const &idata,
                    int const idataCols,
                    std::vector<T> const &ifvalue,
                    int const ifvalueCols,
                    int const nRows);

    /*!
     * \brief Helper function to load the matrix of type TM from a file 
     * 
     * \tparam TM Matrix type
     * 
     * \param MX  Input matrix of data
     *
     * \returns true If no error occurs during reading a matrix
     */
    template <typename TM>
    bool loadMatrix(TM &MX);

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam T  Data type 
     * 
     * \param   idata  Array of input data of type T
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns
     * \param options  (0 default) Load matrix from matrix format and 1 load matrix from vector format
     *
     * \returns true If no error occurs during reading data
     */
    template <typename T>
    bool loadMatrix(T **idata, int const nRows, int const nCols, int const options = 0);

    /*!
     * \brief Helper function to load one matrix (or entries number of matrcies) from a file 
     * 
     * \tparam T  Data type 
     * 
     * \param   idata  Array of input data of type T
     * \param   nRows  Number of rows
     * \param   nCols  Number of columns for each row
     * \param options  (0 Default) Load matrix from matrix format and 1 load matrix from vector format
     * \param  entries Number of data entry
     *
     * \returns true if no error occurs during reading data
     */
    template <typename T>
    bool loadMatrix(T **idata, int const nRows, int const *nCols, int const options = 0, int const entries = 1);

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam T  Data type 
     * 
     * \param idata  Input array of data of type T
     * \param nRows  Number of rows
     * \param nCols  Number of columns for each row (default is 1)
     *
     * \returns true if no error occurs during reading data
     */
    template <typename T>
    bool loadMatrix(T *idata, int const nRows, int const nCols = 1);

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam T  Data type 
     * 
     * \param idata  Input array of data of type T
     * \param nRows  Number of rows
     * \param nCols  Number of columns for each row (default is 1)
     *
     * \returns true if no error occurs during reading data
     */
    template <typename T>
    bool loadMatrix(std::unique_ptr<T[]> &idata, int const nRows, int const nCols = 1);

    /*!
     * \brief Helper function to load the matrix from a file 
     * 
     * \tparam T  Data type 
     * 
     * \param idata  Input array of data of type T
     * \param nRows  Number of rows
     * \param nCols  Number of columns for each row (default is 1)
     *
     * \returns true if no error occurs during reading data
     */
    template <typename T>
    bool loadMatrix(std::vector<T> &idata, int const nRows, int const nCols = 1);

    /*!
     * \brief Helper function to load two vector of data from a file
     * 
     * \tparam T Data type 
     * 
     * \param idata        Input array of data of type T
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type T
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows in each data set
     * 
     * \returns true If no error occurs during reading data
     */
    template <typename T>
    bool loadMatrix(T *idata, int const idataCols, T *ifvalue, int const ifvalueCols, int const nRows);

    /*!
     * \brief Helper function to load two vector of data from a file
     * 
     * \tparam T Data type 
     * 
     * \param idata        Input array of data of type T
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type T
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows in each data set
     * 
     * \returns true If no error occurs during reading data
     */
    template <typename T>
    bool loadMatrix(std::unique_ptr<T[]> &idata, int const idataCols, std::unique_ptr<T[]> &ifvalue, int const ifvalueCols, int const nRows);

    /*!
     * \brief Helper function to load two vector of data from a file
     * 
     * \tparam T Data type 
     * 
     * \param idata        Input array of data of type T
     * \param idataCols    Number of columns of inpput array data (idata)
     * \param ifvalue      Array of input value data of type T
     * \param ifvalueCols  Number of columns of inpput value data (ifvalue)
     * \param nRows        Number of rows in each data set
     * 
     * \returns true If no error occurs during reading data
     */
    template <typename T>
    bool loadMatrix(std::vector<T> &idata, int const idataCols, std::vector<T> &ifvalue, int const ifvalueCols, int const nRows);

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param  title       Title (string) that should be written at the top 
     * \param  idata       Array of input data of type T
     * \param  nRows       Number of rows
     * \param  nCols       Number of columns
     * \param  printPrefix Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(const char *title,
                     T const **idata,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param idata        Array of input data of type T
     * \param nRows        Number of rows
     * \param nCols        Number of columns
     * \param  printPrefix Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(T const **idata,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param idata  Array of input data of type T
     * \param nRows  Number of rows
     * \param nCols  Number of columns
     * \param form   Print format
     */
    template <typename T>
    void printMatrix(T const **idata,
                     int const nRows,
                     int const nCols,
                     ioFormat const &form);

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param title       Title (string) that should be written at the top 
     * \param idata       Array of input data of type T
     * \param nRows       Number of rows
     * \param nCols       Number of columns for each row
     * \param entries     Number of data entry   
     * \param printPrefix Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(const char *title,
                     T const **idata,
                     int const nRows,
                     int const *nCols,
                     int const entries = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param idata       Array of input data of type T
     * \param nRows       Number of rows
     * \param nCols       Number of columns for each row
     * \param printPrefix Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(T const **idata,
                     int const nRows,
                     int const *nCols,
                     int const entries = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print one matrix (or entries number of matrices)
     * 
     * \tparam T Data type
     * 
     * \param idata   Array of input data of type T
     * \param nRows   Number of rows
     * \param nCols   Number of columns for each row
     * \param entries Number of data entry   
     * \param form    Print format for each row 
     */
    template <typename T>
    void printMatrix(T const **idata,
                     int const nRows,
                     int const *nCols,
                     int const entries = 1,
                     std::vector<ioFormat> const &form = std::vector<ioFormat>());

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param title       Title (string) that should be written at the top 
     * \param idata       Array of input data of type T
     * \param nRows       Number of rows
     * \param nCols       Number of columns (default is 1)
     * \param printPrefix Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(const char *title,
                     T const *idata,
                     int const nRows,
                     int const nCols = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    template <typename T>
    void printMatrix(const char *title,
                     std::unique_ptr<T[]> const &idata,
                     int const nRows,
                     int const nCols = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    template <typename T>
    void printMatrix(const char *title,
                     std::vector<T> const &idata,
                     int const nRows,
                     int const nCols = 1,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param idata  Array of input data of type T
     * \param nRows  Number of rows
     * \param nCols  Number of columns (default is 1)
     * \param form   Print format
     */
    template <typename T>
    void printMatrix(T const *idata,
                     int const nRows,
                     int const nCols = 1,
                     ioFormat const &form = ioFormat("NO"));

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param idata  Array of input data of type T
     * \param nRows  Number of rows
     * \param nCols  Number of columns (default is 1)
     * \param form   Print format
     */
    template <typename T>
    void printMatrix(std::unique_ptr<T[]> const &idata,
                     int const nRows,
                     int const nCols = 1,
                     ioFormat const &form = ioFormat("NO"));

    /*!
     * \brief Helper function to print the matrix
     * 
     * \tparam T Data type
     * 
     * \param idata  Array of input data of type T
     * \param nRows  Number of rows
     * \param nCols  Number of columns (default is 1)
     * \param form   Print format
     */
    template <typename T>
    void printMatrix(std::vector<T> const &idata,
                     int const nRows,
                     int const nCols = 1,
                     ioFormat const &form = ioFormat("NO"));

    /*!
     * \brief Helper function to print one element of input data
     * 
     * \tparam T Data type
     * 
     * \param idata  Array of input data of type T
     * \param form   Print format
     */
    template <typename T>
    void printMatrix(T const *idata, ioFormat const &form);

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam T Data type
     * 
     * \param title          Title (string) that should be written at the top 
     * \param idata          Array of input data of type T
     * \param idataCols      Number of columns of inpput array data (idata)
     * \param ifvalue        Array of input value data of type T
     * \param ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param nRows          Number of rows
     * \param printPrefix    Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(const char *title,
                     T const *idata,
                     int const idataCols,
                     T const *ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam T Data type
     * 
     * \param title          Title (string) that should be written at the top 
     * \param idata          Array of input data of type T
     * \param idataCols      Number of columns of inpput array data (idata)
     * \param ifvalue        Array of input value data of type T
     * \param ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param nRows          Number of rows
     * \param printPrefix    Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(const char *title,
                     std::unique_ptr<T[]> const &idata,
                     int const idataCols,
                     std::unique_ptr<T[]> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam T Data type
     * 
     * \param title          Title (string) that should be written at the top 
     * \param idata          Array of input data of type T
     * \param idataCols      Number of columns of inpput array data (idata)
     * \param ifvalue        Array of input value data of type T
     * \param ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param nRows          Number of rows
     * \param printPrefix    Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(const char *title,
                     std::vector<T> const &idata,
                     int const idataCols,
                     std::vector<T> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam T Data type
     * 
     * \param title          Title (string) that should be written at the top 
     * \param idata          Array of input data of type T
     * \param idataCols      Number of columns of inpput array data (idata)
     * \param ifvalue        Array of input value data of type T
     * \param ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param nRows          Number of rows
     * \param printPrefix    Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(T const *idata,
                     int const idataCols, T *ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam T Data type
     * 
     * \param title          Title (string) that should be written at the top 
     * \param idata          Array of input data of type T
     * \param idataCols      Number of columns of inpput array data (idata)
     * \param ifvalue        Array of input value data of type T
     * \param ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param nRows          Number of rows
     * \param printPrefix    Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(std::unique_ptr<T[]> const &idata,
                     int const idataCols,
                     std::unique_ptr<T[]> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data
     * 
     * \tparam T Data type
     * 
     * \param title          Title (string) that should be written at the top 
     * \param idata          Array of input data of type T
     * \param idataCols      Number of columns of inpput array data (idata)
     * \param ifvalue        Array of input value data of type T
     * \param ifvalueCols    Number of columns of inpput value data (ifvalue)
     * \param nRows          Number of rows
     * \param printPrefix    Prefix and suffix of the print  
     */
    template <typename T>
    void printMatrix(std::vector<T> const &idata,
                     int const idataCols,
                     std::vector<T> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix = "\n----------------------------------------\n");

    /*!
     * \brief Helper function to print two vectors of data with stream format for each
     * 
     * \tparam T Data type
     * 
     * \param idata         Array of input data of type T
     * \param idataCols     Number of columns of inpput array data (idata)
     * \param ifvalue       Array of input value data of type T
     * \param ifvalueCols   Number of columns of inpput value data (ifvalue)
     * \param nRows         Number of rows
     * \param formD         Print format for input data
     * \param formF         Print format for input function value 
     */
    template <typename T>
    void printMatrix(T const *idata,
                     int const idataCols,
                     T const *ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     ioFormat const &formD,
                     ioFormat const &formF);

    /*!
     * \brief Helper function to print two vectors of data with stream format for each
     * 
     * \tparam T Data type
     * 
     * \param idata         Array of input data of type T
     * \param idataCols     Number of columns of inpput array data (idata)
     * \param ifvalue       Array of input value data of type T
     * \param ifvalueCols   Number of columns of inpput value data (ifvalue)
     * \param nRows         Number of rows
     * \param formD         Print format for input data
     * \param formF         Print format for input function value 
     */
    template <typename T>
    void printMatrix(std::unique_ptr<T[]> const &idata,
                     int const idataCols,
                     std::unique_ptr<T[]> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     ioFormat const &formD,
                     ioFormat const &formF);

    /*!
     * \brief Helper function to print two vectors of data with stream format for each
     * 
     * \tparam T Data type
     * 
     * \param idata         Array of input data of type T
     * \param idataCols     Number of columns of inpput array data (idata)
     * \param ifvalue       Array of input value data of type T
     * \param ifvalueCols   Number of columns of inpput value data (ifvalue)
     * \param nRows         Number of rows
     * \param formD         Print format for input data
     * \param formF         Print format for input function value 
     */
    template <typename T>
    void printMatrix(std::vector<T> const &idata,
                     int const idataCols,
                     std::vector<T> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     ioFormat const &formD,
                     ioFormat const &formF);

  private:
    //! Input/output file based stream
    std::fstream fs;

    //! Line for reading the string of data
    std::string line;

    //! Index type
    typedef std::ptrdiff_t Idx;

    //! Width parameter of the stream out or in
    std::ptrdiff_t Width;

    //! Flag for using the pre defined Width or computing it on the fly from the input data
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

template <typename T>
inline void io::setPrecision(std::ostream &streamBuffer)
{
    if (std::numeric_limits<T>::is_integer)
    {
        // Manages the precision (i.e. how many digits are generated)
        streamBuffer.precision(0);
    }
    else
    {
        // Manages the precision (i.e. how many digits are generated)
        streamBuffer.precision(digits10<T>());
        streamBuffer << std::fixed;
    }
}

inline void io::setWidth(int streamWidth)
{
    Width = streamWidth < 0 ? std::ptrdiff_t{} : static_cast<std::ptrdiff_t>(streamWidth);
    FixedWidth = streamWidth >= 0;
}

template <typename T>
int io::getWidth(T *idata, int const nRows, int const nCols, std::ostream &streamBuffer)
{
    std::ptrdiff_t tWidth(0);
    setPrecision<T>(streamBuffer);
    for (int i = 0; i < nRows * nCols; i++)
    {
        std::stringstream sstr;
        sstr.copyfmt(streamBuffer);
        sstr << idata[i];
        tWidth = std::max<std::ptrdiff_t>(tWidth, io::Idx(sstr.str().length()));
    }
    return static_cast<int>(tWidth);
}

template <typename T>
int io::getWidth(std::unique_ptr<T[]> const &idata, int const nRows, int const nCols, std::ostream &streamBuffer)
{
    std::ptrdiff_t tWidth(0);
    setPrecision<T>(streamBuffer);
    for (int i = 0; i < nRows * nCols; i++)
    {
        std::stringstream sstr;
        sstr.copyfmt(streamBuffer);
        sstr << idata[i];
        tWidth = std::max<std::ptrdiff_t>(tWidth, io::Idx(sstr.str().length()));
    }
    return static_cast<int>(tWidth);
}

template <typename T>
int io::getWidth(std::vector<T> const &idata, int const nRows, int const nCols, std::ostream &streamBuffer)
{
    std::ptrdiff_t tWidth(0);
    setPrecision<T>(streamBuffer);
    for (int i = 0; i < nRows * nCols; i++)
    {
        std::stringstream sstr;
        sstr.copyfmt(streamBuffer);
        sstr << idata[i];
        tWidth = std::max<std::ptrdiff_t>(tWidth, io::Idx(sstr.str().length()));
    }
    return static_cast<int>(tWidth);
}

template <typename T>
int io::getWidth(T **idata, int const nRows, int const nCols, std::ostream &streamBuffer)
{
    std::ptrdiff_t tWidth(0);
    setPrecision<T>(streamBuffer);
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            std::stringstream sstr;
            sstr.copyfmt(streamBuffer);
            sstr << idata[i][j];
            tWidth = std::max<std::ptrdiff_t>(tWidth, io::Idx(sstr.str().length()));
        }
    }
    return static_cast<int>(tWidth);
}

template <typename TM, typename TF>
bool io::saveMatrix(TM &MX, TF const &IOfmt)
{
    if (fs.is_open())
    {
        fs << std::fixed;
        fs << MX.format(IOfmt);
        fs << fmt.rowSeparator;

        return true;
    }
    UMUQFAILRETURN("This file stream is not open for writing!");
}

template <typename T>
bool io::saveMatrix(T **idata, int const nRows, int const nCols, int const options)
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

    setPrecision<T>(fs);

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

template <typename T>
bool io::saveMatrix(T **idata,
                    int const nRows,
                    int const *nCols,
                    int const options,
                    int const entries,
                    std::vector<ioFormat> const &form)
{
    if (!fs.is_open())
    {
        UMUQFAILRETURN("This file stream is not open for writing!");
    }

    setPrecision<T>(fs);

    //Default case, only one set of data
    if (entries == 1)
    {
        if (form.size() != static_cast<decltype(form.size())>(nRows))
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
                        Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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
        if (form.size() != static_cast<decltype(form.size())>(nRows))
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
                    T *ePointer = idata[i];
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
                    T *ePointer;
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
                    T *ePointer;
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
                    }
                }
            }
            else
            {
                if (options == 1)
                {
                    T *ePointer;
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
                else
                {
                    T *ePointer;
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
                    T *ePointer = idata[i];
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
                T *ePointer;
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
                T *ePointer;
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

template <typename T>
bool io::saveMatrix(T *idata,
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

    setPrecision<T>(fs);

    if (!FixedWidth)
    {
        Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

        for (int i = 0; i < nRows * nCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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
                fs << idata[l++];
                for (int j = 1; j < nCols; j++)
                {
                    fs << fmt.coeffSeparator;
                    fs.width(Width);
                    fs << idata[l++];
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
                fs << idata[l++];
                for (int j = 1; j < nCols; j++)
                {
                    fs << fmt.coeffSeparator;
                    fs << idata[l++];
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

template <typename T>
bool io::saveMatrix(std::unique_ptr<T[]> const &idata,
                    int const nRows,
                    int const nCols,
                    int const options)
{
    return io::saveMatrix<T>(idata.get(), nRows, nCols, options);
}

template <typename T>
bool io::saveMatrix(std::vector<T> const &idata,
                    int const nRows,
                    int const nCols,
                    int const options)
{
    return io::saveMatrix<T>(const_cast<T *>(idata.data()), nRows, nCols, options);
}

template <typename T>
bool io::saveMatrix(T *idata,
                    int const idataCols,
                    T *ifvalue,
                    int const ifvalueCols,
                    int const nRows)
{
    if (!fs.is_open())
    {
        UMUQFAILRETURN("This file stream is not open for writing!");
    }

    setPrecision<T>(fs);

    if (!FixedWidth)
    {
        Width = fs.tellp() == 0 ? 0 : std::max<std::ptrdiff_t>(0, Width);

        for (int i = 0; i < nRows * idataCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }

        for (int i = 0; i < nRows * ifvalueCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(fs);
            sstr << ifvalue[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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

template <typename T>
bool io::saveMatrix(std::unique_ptr<T[]> const &idata,
                    int const idataCols,
                    std::unique_ptr<T[]> const &ifvalue,
                    int const ifvalueCols,
                    int const nRows)
{
    return io::saveMatrix<T>(idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows);
}

template <typename T>
bool io::saveMatrix(std::vector<T> const &idata,
                    int const idataCols,
                    std::vector<T> const &ifvalue,
                    int const ifvalueCols,
                    int const nRows)
{
    return io::saveMatrix<T>(const_cast<T *>(idata.data()), idataCols, ifvalue.data(), ifvalueCols, nRows);
}

template <typename TM>
bool io::loadMatrix(TM &MX)
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

template <typename T>
bool io::loadMatrix(T **idata, int const nRows, int const nCols, int const options)
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

template <typename T>
bool io::loadMatrix(T **idata, int const nRows, int const *nCols, int const options, int const entries)
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
            T *ePointer;
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
            T *ePointer;
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
            }
            return true;
        }
        return false;
    }
}

template <typename T>
bool io::loadMatrix(T *idata, int const nRows, int const nCols)
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

                for (int j = 0; j < nCols; j++)
                {
                    inLine >> idata[l++];
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

template <typename T>
bool io::loadMatrix(std::unique_ptr<T[]> &idata, int const nRows, int const nCols)
{
    return io::loadMatrix<T>(idata.get(), nRows, nCols);
}

template <typename T>
bool io::loadMatrix(std::vector<T> &idata, int const nRows, int const nCols)
{
    return io::loadMatrix<T>(idata.data(), nRows, nCols);
}

template <typename T>
bool io::loadMatrix(T *idata, int const idataCols, T *ifvalue, int const ifvalueCols, int const nRows)
{
    std::string Line;

    for (int i = 0, k = 0, l = 0; i < nRows; i++)
    {
        if (std::getline(fs, Line))
        {
            std::stringstream inLine(Line);

            for (int j = 0; j < idataCols; j++)
            {
                inLine >> idata[k++];
            }

            for (int j = 0; j < ifvalueCols; j++)
            {
                inLine >> ifvalue[l++];
            }
        }
        else
        {
            return false;
        }
    }
    return true;
}

template <typename T>
bool io::loadMatrix(std::unique_ptr<T[]> &idata,
                    int const idataCols,
                    std::unique_ptr<T[]> &ifvalue,
                    int const ifvalueCols,
                    int const nRows)
{
    return io::loadMatrix<T>(idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows);
}

template <typename T>
bool io::loadMatrix(std::vector<T> &idata,
                    int const idataCols,
                    std::vector<T> &ifvalue,
                    int const ifvalueCols,
                    int const nRows)
{
    return io::loadMatrix<T>(idata.data(), idataCols, ifvalue.data(), ifvalueCols, nRows);
}

template <typename T>
void io::printMatrix(const char *title,
                     T const **idata,
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

    setPrecision<T>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(std::cout);
                sstr << idata[i][j];
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

template <typename T>
void io::printMatrix(T const **idata,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix)
{
    io::printMatrix<T>("", idata, nRows, nCols, printPrefix);
}

template <typename T>
void io::printMatrix(T const **idata,
                     int const nRows,
                     int const nCols,
                     ioFormat const &form)
{
    setPrecision<T>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                std::stringstream sstr;
                sstr.copyfmt(std::cout);
                sstr << idata[i][j];
                Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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

template <typename T>
void io::printMatrix(const char *title,
                     T const **idata,
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

    setPrecision<T>(std::cout);

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
                T *ePointer = idata[i];
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
            T *ePointer;
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
            T *ePointer;
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

template <typename T>
void io::printMatrix(T const **idata,
                     int const nRows,
                     int const *nCols,
                     int const entries,
                     std::string const &printPrefix)
{
    io::printMatrix<T>("", idata, nRows, nCols, entries, printPrefix);
}

template <typename T>
void io::printMatrix(T const **idata,
                     int const nRows,
                     int const *nCols,
                     int const entries,
                     std::vector<ioFormat> const &form)
{
    if (form.size() != static_cast<decltype(form.size())>(nRows))
    {
        io::printMatrix<T>(idata, nRows, nCols, entries);
    }
    else
    {
        setPrecision<T>(std::cout);

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
                        Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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
                    T *ePointer = idata[i];
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
                T *ePointer;
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
                T *ePointer;
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

template <typename T>
void io::printMatrix(const char *title,
                     T const *idata,
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

    setPrecision<T>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows * nCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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
                std::cout << idata[l++];
                for (int j = 1; j < nCols; j++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout.width(Width);
                    std::cout << idata[l++];
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
                std::cout << idata[l++];
                for (int j = 1; j < nCols; j++)
                {
                    std::cout << fmt.coeffSeparator;
                    std::cout << idata[l++];
                }
                std::cout << fmt.rowSuffix;
                std::cout << fmt.rowSeparator;
            }
        }
    }
    std::cout << printPrefix;
}

template <typename T>
void io::printMatrix(const char *title,
                     std::unique_ptr<T[]> const &idata,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix)
{
    io::printMatrix<T>(title, idata.get(), nRows, nCols, printPrefix);
}

template <typename T>
void io::printMatrix(const char *title,
                     std::vector<T> const &idata,
                     int const nRows,
                     int const nCols,
                     std::string const &printPrefix)
{
    io::printMatrix<T>(title, idata.data(), nRows, nCols, printPrefix);
}

template <typename T>
void io::printMatrix(T const *idata,
                     int const nRows,
                     int const nCols,
                     ioFormat const &form)
{
    if (form.coeffSeparator == "NO")
    {
        io::printMatrix<T>("", idata, nRows, nCols);
    }
    else
    {
        setPrecision<T>(std::cout);

        if (!FixedWidth)
        {
            for (int i = 0; i < nRows * nCols; i++)
            {
                std::stringstream sstr;
                sstr.copyfmt(std::cout);
                sstr << idata[i];
                Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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
                    std::cout << idata[l++];
                    for (int j = 1; j < nCols; j++)
                    {
                        std::cout << form.coeffSeparator;
                        std::cout.width(Width);
                        std::cout << idata[l++];
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
                    std::cout << idata[l++];
                    for (int j = 1; j < nCols; j++)
                    {
                        std::cout << form.coeffSeparator;
                        std::cout << idata[l++];
                    }
                    std::cout << form.rowSuffix;
                    std::cout << form.rowSeparator;
                }
            }
        }
    }
}

template <typename T>
void io::printMatrix(std::unique_ptr<T[]> const &idata,
                     int const nRows,
                     int const nCols,
                     ioFormat const &form)
{
    io::printMatrix<T>(idata.get(), nRows, nCols, form);
}

template <typename T>
void io::printMatrix(std::vector<T> const &idata,
                     int const nRows,
                     int const nCols,
                     ioFormat const &form)
{
    io::printMatrix<T>(idata.data(), nRows, nCols, form);
}

template <typename T>
void io::printMatrix(T const *idata, ioFormat const &form)
{
    setPrecision<T>(std::cout);

    if (!FixedWidth)
    {
        std::stringstream sstr;
        sstr.copyfmt(std::cout);
        sstr << *idata;
        Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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

template <typename T>
void io::printMatrix(const char *title,
                     T const *idata,
                     int const idataCols,
                     T const *ifvalue,
                     int const ifvalueCols,
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

    setPrecision<T>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows * idataCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }

        for (int i = 0; i < nRows * ifvalueCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << ifvalue[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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

template <typename T>
void io::printMatrix(const char *title,
                     std::unique_ptr<T[]> const &idata,
                     int const idataCols,
                     std::unique_ptr<T[]> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<T>(title, idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows, printPrefix);
}

template <typename T>
void io::printMatrix(const char *title,
                     std::vector<T> const &idata,
                     int const idataCols,
                     std::vector<T> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<T>(title, idata.data(), idataCols, ifvalue.data(), ifvalueCols, nRows, printPrefix);
}

template <typename T>
void io::printMatrix(T const *idata,
                     int const idataCols, T *ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<T>("", idata, idataCols, ifvalue, ifvalueCols, nRows, printPrefix);
}

template <typename T>
void io::printMatrix(std::unique_ptr<T[]> const &idata,
                     int const idataCols,
                     std::unique_ptr<T[]> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<T>("", idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows, printPrefix);
}

template <typename T>
void io::printMatrix(std::vector<T> const &idata,
                     int const idataCols,
                     std::vector<T> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     std::string const &printPrefix)
{
    io::printMatrix<T>("", idata.data(), idataCols, ifvalue.data(), ifvalueCols, nRows, printPrefix);
}

template <typename T>
void io::printMatrix(T const *idata,
                     int const idataCols,
                     T const *ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     ioFormat const &formD,
                     ioFormat const &formF)
{
    setPrecision<T>(std::cout);

    if (!FixedWidth)
    {
        for (int i = 0; i < nRows * idataCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << idata[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
        }

        for (int i = 0; i < nRows * ifvalueCols; i++)
        {
            std::stringstream sstr;
            sstr.copyfmt(std::cout);
            sstr << ifvalue[i];
            Width = std::max<std::ptrdiff_t>(Width, io::Idx(sstr.str().length()));
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

template <typename T>
void io::printMatrix(std::unique_ptr<T[]> const &idata,
                     int const idataCols,
                     std::unique_ptr<T[]> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     ioFormat const &formD,
                     ioFormat const &formF)
{
    io::printMatrix<T>(idata.get(), idataCols, ifvalue.get(), ifvalueCols, nRows, formD, formF);
}

template <typename T>
void io::printMatrix(std::vector<T> const &idata,
                     int const idataCols,
                     std::vector<T> const &ifvalue,
                     int const ifvalueCols,
                     int const nRows,
                     ioFormat const &formD,
                     ioFormat const &formF)
{
    io::printMatrix<T>(idata.data(), idataCols, ifvalue.data(), ifvalueCols, nRows, formD, formF);
}

} // namespace umuq

#endif // UMUQ_IO
