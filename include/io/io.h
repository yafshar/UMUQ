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
    inline bool operator==(ioFormat const &rhs);

    inline bool operator==(ioFormat const &rhs) const;

    /*!
     * \brief Operator !=
     * 
     * \param rhs 
     * \return true 
     * \return false 
     */
    inline bool operator!=(ioFormat const &rhs);

    inline bool operator!=(ioFormat const &rhs) const;
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
    inline bool isFileOpened() const;

    /*!
     * \brief Check to see whether the file fileName exists and accessible to read or write!
     *  
     * \returns true if the file exists 
     */
    inline bool isFileExist(const char *fileName);

    inline bool isFileExist(std::string const &fileName);

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
    bool openFile(const char *fileName, const std::ios_base::openmode mode = in);
    bool openFile(std::string const &fileName, const std::ios_base::openmode mode = in);

    /*!
     * \brief Get string from stream
     * 
     * Get a string from stream and stores them into line until 
     * a newline or the end-of-file is reached, whichever happens first.
     * 
     * \returns true if no error occurs on the associated stream
     */
    bool readLine(const char comment = '#');

    /*!
     * \brief Set position of stream to the beginning
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
     * \return std::string& 
     */
    inline std::string &getLine();

    /*!
     * \brief Set the stream Precision 
     * 
     * \tparam TD Data type
     * 
     * \param  os File based streams
     */
    template <typename TD>
    inline void setPrecision(std::ostream &os);

    /*!
     * \brief Set the width parameter of the stream to exactly n.
     * 
     * If Input Width_ is < 0 the function will set the stream to zero and its setting flag to false 
     * 
     * \param Width_ New value for Width 
     */
    inline void setWidth(int Width_ = 0);

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
    int getWidth(TD *idata, int const nRows, int const nCols, std::ostream &os);

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
    int getWidth(std::unique_ptr<TD[]> const &idata, int const nRows, int const nCols, std::ostream &os);

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
    int getWidth(TD **idata, int const nRows, int const nCols, std::ostream &os);

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
    bool saveMatrix(TD **idata, int const nRows, int const nCols, int const options = 0);

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
    bool saveMatrix(TD **idata, int const nRows, int const *nCols, int const options = 0, int const entries = 1, std::vector<ioFormat> const &form = std::vector<ioFormat>());

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
    bool saveMatrix(TD *idata, int const nRows, int const nCols = 1, int const options = 0);

    template <typename TD>
    bool saveMatrix(std::unique_ptr<TD[]> const &idata, int const nRows, int const nCols = 1, int const options = 0);

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
    bool saveMatrix(TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows);

    template <typename TD>
    bool saveMatrix(std::unique_ptr<TD[]> const &idata, int const idataCols, std::unique_ptr<TD[]> const &ifvalue, int const ifvalueCols, int const nRows);


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
    bool loadMatrix(TM &MX);


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
    bool loadMatrix(TD **idata, int const nRows, int const nCols, int const options = 0);

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
    bool loadMatrix(TD **idata, int const nRows, int const *nCols, int const options = 0, int const entries = 1);

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
    bool loadMatrix(TD *idata, int const nRows, int const nCols = 1);

    template <typename TD>
    bool loadMatrix(std::unique_ptr<TD[]> &idata, int const nRows, int const nCols = 1);

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
    bool loadMatrix(TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows);

    template <typename TD>
    bool loadMatrix(std::unique_ptr<TD[]> &idata, int const idataCols, std::unique_ptr<TD[]> &ifvalue, int const ifvalueCols, int const nRows);

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
    void printMatrix(const char *title, TD **idata, int const nRows, int const nCols, std::string const &printPrefix = "\n----------------------------------------\n");

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
    void printMatrix(TD **idata, int const nRows, int const nCols, std::string const &printPrefix = "\n----------------------------------------\n");

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
    void printMatrix(TD **idata, int const nRows, int const nCols, ioFormat const &form);

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
    void printMatrix(const char *title, TD **idata, int const nRows, int const *nCols, int const entries = 1, std::string const &printPrefix = "\n----------------------------------------\n");

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
    void printMatrix(TD **idata, int const nRows, int const *nCols, int const entries = 1, std::string const &printPrefix = "\n----------------------------------------\n");

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
    void printMatrix(TD **idata, int const nRows, int const *nCols, int const entries = 1, std::vector<ioFormat> const &form = std::vector<ioFormat>());

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
    void printMatrix(const char *title, TD *idata, int const nRows, int const nCols = 1, std::string const &printPrefix = "\n----------------------------------------\n");

    template <typename TD>
    void printMatrix(const char *title, std::unique_ptr<TD[]> const &idata, int const nRows, int const nCols = 1, std::string const &printPrefix = "\n----------------------------------------\n");

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
    void printMatrix(TD *idata, int const nRows, int const nCols = 1, ioFormat const &form = ioFormat("NO"));

    template <typename TD>
    void printMatrix(std::unique_ptr<TD[]> const &idata, int const nRows, int const nCols = 1, ioFormat const &form = ioFormat("NO"));

    /*!
     * \brief Helper function to print one element of input data
     * 
     * \tparam  TD    Data type
     * 
     * \param  idata  Array of input data of type TD
     * \param  form   Print format
     */
    template <typename TD>
    void printMatrix(TD *idata, ioFormat const &form);

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
    void printMatrix(const char *title, TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows, std::string const &printPrefix = "\n----------------------------------------\n");

    template <typename TD>
    void printMatrix(const char *title, std::unique_ptr<TD[]> const &idata, int const idataCols, std::unique_ptr<TD[]> const &ifvalue, int const ifvalueCols, int const nRows, std::string const &printPrefix = "\n----------------------------------------\n");

    template <typename TD>
    void printMatrix(TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows, std::string const &printPrefix = "\n----------------------------------------\n");

    template <typename TD>
    void printMatrix(std::unique_ptr<TD[]> const &idata, int const idataCols, std::unique_ptr<TD[]> const &ifvalue, int const ifvalueCols, int const nRows, std::string const &printPrefix = "\n----------------------------------------\n");

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
    void printMatrix(TD *idata, int const idataCols, TD *ifvalue, int const ifvalueCols, int const nRows, ioFormat const &formD, ioFormat const &formF);

    template <typename TD>
    void printMatrix(std::unique_ptr<TD[]> const &idata, int const idataCols, std::unique_ptr<TD[]> const &ifvalue, int const ifvalueCols, int const nRows, ioFormat const &formD, ioFormat const &formF);

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
