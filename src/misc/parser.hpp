#ifndef UMUQ_PARSER_H
#define UMUQ_PARSER_H

namespace umuq
{

/*! \class parser
 * \brief  This class prases string of data to seperate words
 * 
 * It ignores all white spaces, tabs, \f$ : \f$ and \f$ , \f$ characters
 */
class parser
{
  public:
    /*!
     * \brief Construct a new parser object
     * 
     */
    parser();

    /*!
     * \brief Destroy the parser object
     * 
     */
    ~parser();

    /*!
     * \brief Parses the input line into tokens
     * 
     * First, it traverses all white spaces, tabs, \f$ : \f$ and \f$ , \f$ characters until 
     * it hits different character which indicates the beginning of an argument.  
     * It saves the address to argv[], and increase the line argument number and skips
     * all characters of this argument. 
     * 
     * \param iline Input line
     */
    void parse(std::string const &iline);

    /*!
     * \brief Parses the input line into tokens
     * 
     * First, it traverses all white spaces, tabs, \f$ : \f$ and \f$ , \f$ characters until 
     * it hits different character which indicates the beginning of an argument.  
     * It saves the address to argv[], and increase the line argument number and skips
     * all characters of this argument. 
     * 
     * \param iline Input line
     */
    void parse(const char *iline);

  private:
    /*!
     * \brief Takes an input line and parse it into tokens
     * 
     * \param iline  Input line
     * \param iargv  Input argv 
     * 
     * 
     * It takes an input line and parse it into tokens.
     * First, it replaces all white spaces with zeros until it hits a non-white space 
     * character which indicates the beginning of an argument.  It saves the address 
     * to argv[], and then skips all non-white spaces which constitute the argument. 
     * 
     * Reference:
     * http://www.csl.mtu.edu/cs4411.ck/www/NOTES/process/fork/shell.c
     * 
     */
    void parse(char *iline, char **iargv);

  public:
    /*!
     * \brief Parses element 
     * 
     * \param  ilineArg  Input string which we want to parse
     * \param  value     Parsed value
     *  
     * \returns Parsed value of type T
     */
    template <typename T>
    inline T &parse(const char *ilineArg, T &value);

    /*!
     * \brief Parses element
     *  
     * \param  ilineArg  Input string which we want to parse
     * 
     * \returns Parsed value of type T
     */
    template <typename T>
    inline T &parse(const char *ilineArg);

    /*!
     * \brief Access element at provided index @id with checking bounds
     * 
     * \param id  Requested index 
     * 
     * \returns Element @(id)
     */
    template <typename T>
    inline T &at(std::size_t const id);

    /*!
     * \brief Access element at provided index @id with no check
     * 
     * param id  Requested id
     * 
     * returns Element @(id)
     */
    template <typename T>
    inline T &operator()(std::size_t const id);

    /*!
     * \brief Access element at provided index @id with no check
     * 
     * param id  Requested id
     * 
     * returns Element @(id)
     */
    template <typename T>
    inline T &operator[](std::size_t const id);

    /*!
     * \brief Get the pointer to lineArg
     * 
     */
    inline char **getLineArg();

    /*!
     * \brief Get the Line Arg object
     * 
     * \param argv         It should have fixed size (char *[])
     * \param LineArgNum   Number of elements in the parsed line
     */
    inline void getLineArg(char **argv, std::size_t &LineArgNum);

  private:
    //! The number of last argument in the parsed line into different words
    std::size_t lineArgNum;

    //! Array of pointers to each word in the parsed line
    char *lineArg[LINESIZE];

    //! Word as an rvalue in parsing string
    std::string svalue;

    //! Temporary string
    std::string lineTmp;
};

parser::parser() : lineArgNum(0) {}

parser::~parser() {}

void parser::parse(std::string const &iline)
{
    char *line = const_cast<char *>(&iline[0]);

    // At the start of parsing each line set the argument number to zero
    lineArgNum = 0;

    // if not the end of line .......
    while (*line != '\0')
    {
        while (*line == ' ' || *line == '\t' || *line == '\n' || *line == ':' || *line == ',')
        {
            line++;
        }

        // Save the argument position
        lineArg[lineArgNum++] = line;

        // Skip the argument until ...
        while (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n' && *line != ':' && *line != ',')
        {
            line++;
        }
    }

    // Mark the end of argument list
    lineArg[lineArgNum] = nullptr;
}

void parser::parse(const char *iline)
{
    char *line = const_cast<char *>(iline);

    // At the start of parsing each line set the argument number to zero
    lineArgNum = 0;

    // if not the end of line .......
    while (*line != '\0')
    {
        while (*line == ' ' || *line == '\t' || *line == '\n' || *line == ':' || *line == ',')
        {
            line++;
        }

        // Save the argument position
        lineArg[lineArgNum++] = line;

        // Skip the argument until ...
        while (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n' && *line != ':' && *line != ',')
        {
            line++;
        }
    }

    // Mark the end of argument list
    lineArg[lineArgNum] = nullptr;
}

void parser::parse(char *iline, char **iargv)
{
    // if not the end of line .......
    while (*iline != '\0')
    {
        while (*iline == ' ' || *iline == '\t' || *iline == '\n' || *iline == ':' || *iline == ',')
        {
            *iline++ = '\0';
        }

        // Save the argument position
        *iargv++ = iline;

        // Skip the argument until ...
        while (*iline != '\0' && *iline != ' ' && *iline != '\t' && *iline != '\n' && *iline != ':' && *iline != ',')
        {
            iline++;
        }
    }

    // Mark the end of argument list
    *iargv = nullptr;
}

template <typename T>
inline T &parser::parse(const char *ilineArg, T &value)
{
    std::stringstream str(ilineArg);
    str >> value;
    return value;
}

template <typename T>
inline T &parser::parse(const char *ilineArg)
{
    T value;
    return parse<T>(ilineArg, value);
}

template <typename T>
inline T &parser::at(std::size_t const id)
{
    if (id >= lineArgNum)
    {
        throw(std::runtime_error("Wrong argument index number!"));
    }

    T rvalue;
    return parse<T>(lineArg[id], rvalue);
}

// Template specialization for string input
template <>
std::string &parser::at<std::string>(std::size_t const id)
{
    if (id >= lineArgNum)
    {
        throw(std::runtime_error("Wrong argument index number!"));
    }
    return parse<std::string>(parser::lineArg[id], parser::svalue);
}

template <typename T>
inline T &parser::operator()(std::size_t const id)
{
    T rvalue;
    return parse<T>(lineArg[id], rvalue);
}

// Template specialization for string input
template <>
std::string &parser::operator()<std::string>(std::size_t const id)
{
    return parse<std::string>(parser::lineArg[id], parser::svalue);
}

template <typename T>
inline T &parser::operator[](std::size_t const id)
{
    T rvalue;
    return parse<T>(lineArg[id], rvalue);
}

// Template specialization for string input
template <>
std::string &parser::operator[]<std::string>(std::size_t const id)
{
    return parse<std::string>(parser::lineArg[id], parser::svalue);
}

inline char **parser::getLineArg()
{
    return lineArg;
}

inline void parser::getLineArg(char **argv, std::size_t &LineArgNum)
{
    LineArgNum = lineArgNum;
    lineTmp = std::string(lineArg[0]);
    parse(const_cast<char *>(lineTmp.c_str()), argv);
}

} // namespace umuq

#endif // UMUQ_PARSER
