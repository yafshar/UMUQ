#ifndef UMHBM_PARSER_H
#define UMHBM_PARSER_H

#define LINESIZE 256

namespace
{
/*! \class parser
*   \brief parser is a class which prase string of data to seperate words
*	
*/
class parser
{
  public:
    /*!
     * \brief parse a line to seperate arguments
     */
    inline void parse(std::string &line_)
    {
        char *line = &line_[0];

        lineNum = 0;

        /* if not the end of line ....... */
        while (*line != '\0')
        {
            while (*line == ' ' || *line == '\t' || *line == '\n' || *line == ':' || *line == ',')
            {
                line++;
            }

            /* save the argument position */
            lineArg[lineNum++] = line;

            /* skip the argument until ...*/
            while (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n' && *line != ':' && *line != ',')
            {
                line++;
            }
        }

        /* mark the end of argument list */
        lineArg[lineNum] = NULL;
    }

    /*!
     * \brief parse a line to seperate arguments
     */
    inline void parse(const char *line_)
    {
        char *line = (char *)line_;
        lineNum = 0;

        /* if not the end of line ....... */
        while (*line != '\0')
        {
            while (*line == ' ' || *line == '\t' || *line == '\n' || *line == ':' || *line == ',')
            {
                line++;
            }

            /* save the argument position */
            lineArg[lineNum++] = line;

            /* skip the argument until ...*/
            while (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n' && *line != ':' && *line != ',')
            {
                line++;
            }
        }

        /* mark the end of argument list */
        lineArg[lineNum] = NULL;
    }

    /*!
     * \brief parse element 
     * @param  lineArg, input string which we want to parse
     * @tparam value 
     * @return value of type T
     */
    template <typename T>
    inline T &parse(const char *lineArg, T &value)
    {
        std::stringstream str(lineArg);
        str >> value;
        return value;
    }

    /*!
     * \brief parse element 
     * @param  lineArg, input string which we want to parse
     * @return value of type T
     */
    template <typename T>
    inline T parse(const char *lineArg)
    {
        T value;
        return parse<T>(lineArg, value);
    }

    /*!
     * \brief access element at provided index
     * @param id
     * @return element @(id)
     */
    template <typename T>
    T &at(size_t id)
    {
        T rvalue;
        return parse<T>(lineArg[id], rvalue);
    }

    /*!
     * \brief access element at provided index 
     * @param id
     * @return element @(id)
     */
    template <typename T>
    T &operator()(size_t id)
    {
        T rvalue;
        return parse<T>(lineArg[id], rvalue);
    }

    /*!
     * \brief Get the pointer to lineArg
     */
    char **getLineArg()
    {
        return lineArg;
    }

  private:
    char *lineArg[LINESIZE];
    size_t lineNum;
    std::string svalue;
};

template <>
std::string &parser::at<std::string>(size_t id)
{
    return parse<std::string>(parser::lineArg[id], parser::svalue);
};

template <>
std::string &parser::operator()<std::string>(size_t id)
{
    return parse<std::string>(parser::lineArg[id], parser::svalue);
}
} //end of namespace

#endif
