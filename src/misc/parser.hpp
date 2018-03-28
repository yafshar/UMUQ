#ifndef UMHBM_PARSER_H
#define UMHBM_PARSER_H

#include <iostream>

//open,fstat
#include <sys/stat.h>
//fopen, fclose
#include <stdio.h>

class parser
{
  public:
    /*!
     *  \brief parse a line to seperate arguments
     */
    inline void parse(char *line, char **argv)
    {
        /* if not the end of line ....... */
        while (*line != '\0')
        {
            while (*line == ' ' || *line == '\t' || *line == '\n' || *line == ':' || *line == ',')
            {
                /* replace white spaces with 0 */
                *line++ = '\0';
            }

            /* save the argument position */
            *argv++ = line;

            /* skip the argument until ...*/
            while (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n' && *line != ':' && *line != ',') 
            {
                line++;
            }
        }

        /* mark the end of argument list */
        *argv = NULL;
    }

    inline void parse(const char *lineArg, int &value)
    {
        sscanf(lineArg, "%d", &value);
    }

    inline void parse(const char *lineArg, long &value)
    {
        sscanf(lineArg, "%ld", &value);
    }

    inline void parse(const char *lineArg, float &value)
    {
        sscanf(lineArg, "%f", &value);
    }

    inline void parse(const char *lineArg, double &value)
    {
        sscanf(lineArg, "%lf", &value);
    }
};

#endif