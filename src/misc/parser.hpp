#ifndef UMUQ_PARSER_H
#define UMUQ_PARSER_H

#if HAVE_PYTHON == 1
#ifdef toupper
#undef toupper
#endif
#ifdef tolower
#undef tolower
#endif
#endif

namespace umuq
{

/*! \class parser
 *
 * \brief This class parses string of data to seperate words
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
     * \param inputLine Input line
     */
	void parse(std::string const &inputLine);

	/*!
     * \brief Parses the input line into tokens
     * 
     * First, it traverses all white spaces, tabs, \f$ : \f$ and \f$ , \f$ characters until 
     * it hits different character which indicates the beginning of an argument.  
     * It saves the address to argv[], and increase the line argument number and skips
     * all characters of this argument. 
     * 
     * \param inputLine Input line
     */
	void parse(const char *inputLine);

private:
	/*!
     * \brief Takes an input line and parse it into tokens
     * 
     * \param inputLine  Input line
     * \param inputArgv  Input vector of arguments 
     * 
     * 
     * It takes an input line and parse it into tokens.
     * First, it replaces all white spaces with zeros until it hits a non-white space 
     * character which indicates the beginning of an argument.  It saves the address 
     * to argv[], and then skips all non-white spaces which constitute the argument. 
     * 
     * Reference:<br>
     * http://www.csl.mtu.edu/cs4411.ck/www/NOTES/process/fork/shell.c
     * 
     */
	void parse(char *inputLine, char **inputArgv);

public:
	/*!
     * \brief Parses element 
     * 
     * \param  inputLineArg  Input string which we want to parse
     * \param  parsedValue   Parsed value
     *  
     * \returns Parsed value of type T
     */
	template <typename T>
	inline T &parse(const char *inputLineArg, T &parsedValue);

	/*!
     * \brief Parses element
     *  
     * \param inputLineArg  Input string which we want to parse
     * 
     * \returns Parsed value of type T
     */
	template <typename T>
	inline T &parse(const char *inputLineArg);

	/*!
     * \brief Access element at provided index id with checking bounds
     * 
     * \param id  Requested index 
     * 
     * \returns Element (id)
     */
	template <typename T>
	inline T &at(std::size_t const id);

	/*!
     * \brief Access element at provided index id with no check
     * 
     * param id  Requested id
     * 
     * returns Element (id)
     */
	template <typename T>
	inline T &operator()(std::size_t const id);

	/*!
     * \brief Access element at provided index id with no check
     * 
     * param id  Requested id
     * 
     * returns Element (id)
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

	/*!
	 * \brief Get the number of line arguments 
	 * 
	 * \returns std::size_t 
	 */
	inline std::size_t getLineArgNum();

	/*!
     * \brief Converts the given string to uppercase according to the 
     * character conversion rules defined by the currently installed C locale. 
     * 
     * \param InputLineArg  Input argument
	 * \param startIndex    The starting index of the character in the string to convert to uppercase
	 * \param endIndex      The ending index of the character in the string to convert to uppercase
     * 
     * \returns Uppercase string 
     */
	inline std::string toupper(std::string const &InputLineArg, std::size_t const startIndex = 0, std::size_t const endIndex = 0);

	/*!
     * \brief Converts the given string to lowercase according to the 
     * character conversion rules defined by the currently installed C locale. 
     * 
     * \param InputLineArg Input argument
	 * \param startIndex   The starting index of the character in the string to convert to lowercase
	 * \param endIndex     The ending index of the character in the string to convert to lowercase
     * 
     * \returns Lowercase string 
     */
	inline std::string tolower(std::string const &InputLineArg, std::size_t const startIndex = 0, std::size_t const endIndex = 0);

private:
	//! The number of last argument in the parsed line into different words
	std::size_t lineArgNum;

	//! Array of pointers to each word in the parsed line
	char *lineArg[LINESIZE];

	//! Word as an rvalue in parsing string
	std::string stringValue;

	//! Temporary string
	std::string lineTmp;
};

parser::parser() : lineArgNum(0) {}

parser::~parser() {}

void parser::parse(std::string const &inputLine)
{
	char *line = const_cast<char *>(&inputLine[0]);

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

void parser::parse(const char *inputLine)
{
	char *line = const_cast<char *>(inputLine);

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

void parser::parse(char *inputLine, char **inputArgv)
{
	// if not the end of line .......
	while (*inputLine != '\0')
	{
		while (*inputLine == ' ' || *inputLine == '\t' || *inputLine == '\n' || *inputLine == ':' || *inputLine == ',')
		{
			*inputLine++ = '\0';
		}

		// Save the argument position
		*inputArgv++ = inputLine;

		// Skip the argument until ...
		while (*inputLine != '\0' && *inputLine != ' ' && *inputLine != '\t' && *inputLine != '\n' && *inputLine != ':' && *inputLine != ',')
		{
			inputLine++;
		}
	}

	// Mark the end of argument list
	*inputArgv = nullptr;
}

template <typename T>
inline T &parser::parse(const char *inputLineArg, T &parsedValue)
{
	std::stringstream str(inputLineArg);
	str >> parsedValue;
	return parsedValue;
}

template <typename T>
inline T &parser::parse(const char *inputLineArg)
{
	T parsedValue;
	return parse<T>(inputLineArg, parsedValue);
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
	return parse<std::string>(parser::lineArg[id], parser::stringValue);
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
	return parse<std::string>(parser::lineArg[id], parser::stringValue);
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
	return parse<std::string>(parser::lineArg[id], parser::stringValue);
}

inline char **parser::getLineArg() { return lineArg; }

inline std::size_t parser::getLineArgNum() { return lineArgNum; }

inline void parser::getLineArg(char **argv, std::size_t &LineArgNum)
{
	LineArgNum = lineArgNum;
	lineTmp = std::string(lineArg[0]);
	parse(const_cast<char *>(lineTmp.c_str()), argv);
}

inline std::string parser::toupper(std::string const &InputLineArg, std::size_t const startIndex, std::size_t const endIndex)
{
	std::string inputLineArg(InputLineArg);
	auto begin = startIndex > 0 ? inputLineArg.begin() + startIndex : inputLineArg.begin();
	auto end = endIndex > 0 ? (endIndex > startIndex) ? inputLineArg.begin() + endIndex : inputLineArg.end() : inputLineArg.end();
	std::transform(begin, end, begin, [](unsigned char c) {
		unsigned char const u = std::toupper(c);
		return (u != c) ? u : c;
	});
	return inputLineArg;
}

inline std::string parser::tolower(std::string const &InputLineArg, std::size_t const startIndex, std::size_t const endIndex)
{
	std::string inputLineArg(InputLineArg);
	auto begin = startIndex > 0 ? inputLineArg.begin() + startIndex : inputLineArg.begin();
	auto end = endIndex > 0 ? (endIndex > startIndex) ? inputLineArg.begin() + endIndex : inputLineArg.end() : inputLineArg.end();
	std::transform(begin, end, begin, [](unsigned char c) {
		unsigned char const l = std::tolower(c);
		return (l != c) ? l : c;
	});
	return inputLineArg;
}

} // namespace umuq

#endif // UMUQ_PARSER
