#ifndef UMHBM_IO_H
#define UMHBM_IO_H

#define _XOPEN_SOURCE 700
#define _BSD_SOURCE 1

#include <iostream>

//fopen, rewind
#include <cstdio>
//strlen
#include <cstring>
//stat
#include <sys/stat.h>

#define LINESIZE 256

/*! \class io
*   \brief io is a class which includes some IO functionality.
*	
*/

class io
{
  public:
    FILE *f;

    char *line;
    char **lineArg;

    io() : f(NULL), line(NULL), lineArg(NULL){};

    ~io()
    {
        closeFile();
    };

    /*!
     *  \brief return true if file is opened
     */
    inline bool isFileOpened() const { return f != NULL; }

    /*!
     *  \brief Check to see whether the file fileName exists and accessible to read or write!
     *  
     */
    inline bool isFileExist(const char *fileName)
    {
        struct stat buffer;
        return (stat(fileName, &buffer) == 0);
    }

    /*!
     *  \brief Opens the file whose name is specified in the parameter filename 
     *  
     *  Opens the file whose name is specified in the parameter filename and
     *  associates it with a stream that can be identified in future operations 
     *  by the FILE pointer returned.inline   
     */
    inline bool openFile(const char *fileName)
    {
        if (!isFileExist(fileName))
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << fileName << " does not exists!" << std::endl;
            return false;
        }

        if (isFileOpened())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Pointer to the File " << fileName << " is busy!" << std::endl;
            return false;
        }

        f = fopen(fileName, "r");

        if (f == NULL)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << fileName << " does not exists!" << std::endl;
            return false;
        }

        try
        {
            line = new char[LINESIZE];
            lineArg = new char *[LINESIZE];
        }
        catch (const std::bad_alloc &e)
        {
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    inline bool openFile(const char *fileName, const char *mode)
    {
        if (*mode != 'r' || isFileExist(fileName))
        {
            if (isFileOpened())
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "Pointer to the File " << fileName << " is busy!" << std::endl;
                return false;
            }

            f = fopen(fileName, mode);
            if (f == NULL)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << fileName << " does not exists!" << std::endl;
                return false;
            }

            try
            {
                line = new char[LINESIZE];
                lineArg = new char *[LINESIZE];
            }
            catch (const std::bad_alloc &e)
            {
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }

            return true;
        }
        else
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << fileName << " does not exists!" << std::endl;
            return false;
        }
    }

    /*!
     *  \brief 
     */
    inline bool readLine() const { return fgets(line, LINESIZE, f) != NULL; }

    /*!
     *  \brief  
     */
    inline bool emptyLine() const { return (line[0] == '#') || (strlen(line) == 0); }

    /*!
     *  \brief 
     */
    inline void rewindFile() { rewind(f); }

    /*!
     *  \brief  close the File
     */
    inline void closeFile()
    {
        if (isFileOpened())
        {
            fclose(f);
            f = NULL;

            delete[] line;
            line = NULL;

            delete[] lineArg;
            lineArg = NULL;
        }
    }
};

#endif