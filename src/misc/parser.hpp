#ifndef UMUQ_PARSER_H
#define UMUQ_PARSER_H

#define LINESIZE 256

/*! \class parser
* \brief This class prases string of data to seperate words
*	
*/
class parser
{
  public:
    /*!
     * \brief Construct a new parser object
     * 
     */
    parser(){}

    /*!
     * \brief Destroy the parser object
     * 
     */
    ~parser() {}

    /*!
     * \brief parse a line to seperate arguments
     */
    void parse(std::string const &line_)
    {
        char *line = const_cast<char *>(&line_[0]);

        //At the start of parsing each line set the argument number to zero
        lineArgNum = 0;

        /* if not the end of line ....... */
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

    /*!
     * \brief parse a line to seperate arguments
     */
    void parse(const char *line_)
    {
        char *line = const_cast<char *>(line_);

        //At the start of parsing each line set the argument number to zero
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

    /*!
     * \brief parse element 
     * 
     * \param  lineArg_ input string which we want to parse
     * \param  value    parsed value 
     * \return value    value of type T
     */
    template <typename T>
    inline T &parse(const char *lineArg_, T &value)
    {
        std::stringstream str(lineArg_);
        str >> value;
        return value;
    }

    /*!
     * \brief parse element
     *  
     * \param  lineArg_ input string which we want to parse
     * \return value of type T
     */
    template <typename T>
    inline T &parse(const char *lineArg_)
    {
        T value;
        return parse<T>(lineArg_, value);
    }

    /*!
     * \brief access element at provided index @id with checking bounds
     * 
     * \param  id requested index 
     * \return element @(id)
     */
    template <typename T>
    inline T &at(std::size_t const id)
    {
        if (id >= lineArgNum)
        {
            throw(std::runtime_error("Wrong argument index number!"));
        }

        T rvalue;
        return parse<T>(lineArg[id], rvalue);
    }

    /*!
     * \brief access element at provided index @id with no check
     * 
     * param id requested id
     * return element @(id)
     */
    template <typename T>
    inline T &operator()(std::size_t const id)
    {
        T rvalue;
        return parse<T>(lineArg[id], rvalue);
    }

    /*!
     * \brief access element at provided index @id with no check
     * 
     * param id requested id
     * return element @(id)
     */
    template <typename T>
    inline T &operator[](std::size_t const id)
    {
        T rvalue;
        return parse<T>(lineArg[id], rvalue);
    }

    /*!
     * \brief Get the pointer to lineArg
     */
    inline char **getLineArg()
    {
        return lineArg;
    }

  private:
    //! Array of pointers to each word in the parsed line
    char *lineArg[LINESIZE];

    //! The number of last argument in the parsed line into different words
    std::size_t lineArgNum;

    //! Word as an rvalue in parsing string
    std::string svalue;
};

//Template specialization for string input
template <>
std::string &parser::at<std::string>(std::size_t const id)
{
    if (id >= lineArgNum)
    {
        throw(std::runtime_error("Wrong argument index number!"));
    }
    return parse<std::string>(parser::lineArg[id], parser::svalue);
}

template <>
std::string &parser::operator()<std::string>(std::size_t const id)
{
    return parse<std::string>(parser::lineArg[id], parser::svalue);
}

template <>
std::string &parser::operator[]<std::string>(std::size_t const id)
{
    return parse<std::string>(parser::lineArg[id], parser::svalue);
}

#endif
