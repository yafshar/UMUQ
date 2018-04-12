#ifndef UMHBM_SPLIT_H
#define UMHBM_SPLIT_H

/*! 
 * \brief split a string on a single delimiter character (delim)
 * 
 */
template <typename T>
T &split(std::string const &line, char const delim, T &words)
{
    std::stringstream line_(line);
    std::string word;
    while (std::getline(line_, word, delim))
    {
        words.push_back(word);
    }
    return words;
}

template <>
std::string &split<std::string>(std::string const &line, char const delim, std::string &words)
{
    std::stringstream line_(line);
    std::getline(line_, words, delim);
    return words;
}

template <typename T>
T split(std::string const &line, char const delim)
{
    T words;
    return split<T>(line, delim, words);
}

/*! 
 * \brief split a string on any character found in the string of delimiters (delims)
 * 
 */
template <>
std::string split<std::string>(std::string const &line, char const delim)
{
    std::string word;
    return split<std::string>(line, delim, word);
}

template <typename T>
T &split(std::string const &line, std::string const &delims, T &words)
{
    char *token;
    char c_chars[line.size() + 1];
    char *c_str = &c_chars[0];
    std::strcpy(c_str, line.c_str());
    token = std::strtok(c_str, delims.c_str());
    while (token != NULL)
    {
        words.push_back(token);
        token = std::strtok(NULL, delims.c_str());
    }
    return words;
}

template <typename T>
T split(std::string const &line, std::string const &delims)
{
    T words;
    return split(line, delims, words);
}

template <typename T>
void test_custom(std::string const &line, char const *d, T &ret)
{
    T output;

    std::bitset<255> delims;
    while (*d)
    {
        unsigned char code = *d++;
        delims[code] = true;
    }

    typedef std::string::const_iterator iter;
    iter beg;
    bool in_token = false;

    for (std::string::const_iterator it = line.begin(), end = line.end(); it != end; ++it)
    {
        if (delims[*it])
        {
            if (in_token)
            {
                output.push_back(typename T::value_type(beg, it));
                in_token = false;
            }
        }
        else if (!in_token)
        {
            beg = it;
            in_token = true;
        }
    }
    if (in_token)
    {
        output.push_back(typename T::value_type(beg, s.end()));
    }

    output.swap(ret);
}

#endif
