#ifndef UMHBM_IO_H
#define UMHBM_IO_H

#define _XOPEN_SOURCE 700
#define _BSD_SOURCE 1

#include <iostream>
#include <fstream>
#include <limits>
#include <ios>

#include <cstdio>     //fopen, rewind
#include <cstring>    //strlen
#include <sys/stat.h> //stat

#define LINESIZE 256

/*! 
  * \brief Stores a set of parameters controlling the way matrices are printed
  *
  * List of available parameters:
  *  - \b coeffSeparator string printed between two coefficients of the same row
  *  - \b rowSeparator string printed between two rows
  *  - \b rowPrefix string printed at the beginning of each row
  *  - \b rowSuffix string printed at the end of each row
  *  - \b matPrefix string printed at the beginning of the matrix
  *  - \b matSuffix string printed at the end of the matrix
  *
  */
struct IOFormat
{
    /** Default constructor, see class IOFormat for the meaning of the parameters */
    IOFormat(const std::string &_coeffSeparator = " ",
             const std::string &_rowSeparator = "\n",
             const std::string &_rowPrefix = "",
             const std::string &_rowSuffix = "",
             const std::string &_matPrefix = "",
             const std::string &_matSuffix = "") : matPrefix(_matPrefix),
                                                   matSuffix(_matSuffix),
                                                   rowPrefix(_rowPrefix),
                                                   rowSuffix(_rowSuffix),
                                                   rowSeparator(_rowSeparator),
                                                   rowSpacer(""),
                                                   coeffSeparator(_coeffSeparator)
    {
        int i = int(matSuffix.length()) - 1;
        while (i >= 0 && matSuffix[i] != '\n')
        {
            rowSpacer += ' ';
            i--;
        }
    }

    std::string coeffSeparator;
    std::string rowSeparator;
    std::string rowPrefix;
    std::string rowSuffix;
    std::string matPrefix;
    std::string matSuffix;
    std::string rowSpacer;
};

/*! \class io
*   \brief io is a class which includes some IO functionality.
*	
*/
class io
{
  public:
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
            std::cerr << "'" << fileName << "' does not exists!" << std::endl;
            return false;
        }

        if (isFileOpened())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Pointer to the File '" << fileName << "' is busy!" << std::endl;
            return false;
        }

        f = fopen(fileName, "r");

        if (f == NULL)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "'" << fileName << "' does not exists!" << std::endl;
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
                std::cerr << "Pointer to the File '" << fileName << "' is busy!" << std::endl;
                return false;
            }

            f = fopen(fileName, mode);
            if (f == NULL)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "'" << fileName << "' does not exists!" << std::endl;
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
            std::cerr << "'" << fileName << "' does not exists!" << std::endl;
            return false;
        }
    }

    /*!
     * \brief Opens the file whose name is specified in the parameter filename 
     *  
     * Opens the file whose name is specified in the parameter filename and
     * associates it with a stream that can be identified in future operations 
     * by the FILE pointer returned.inline   
     * 
     * stream open mode type
     * 
     * std::fstream::app 	  seek to the end of stream before each write
     * std::fstream::binary   open in binary mode
     * std::fstream::in 	  open for reading
     * std::fstream::out 	  open for writing
     * std::fstream::trunc 	  discard the contents of the stream when opening
     * std::fstream::ate 	  seek to the end of stream immediately after open
     */
    inline bool openFile(const char *fileName, std::ios_base::openmode mode)
    {
        if (mode != std::fstream::in || isFileExist(fileName))
        {
            if (fs.is_open())
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "Pointer to the File '" << fileName << "' is busy!" << std::endl;
                return false;
            }

            fs.open(fileName, mode);
            if (!fs.is_open())
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "'" << fileName << "' does not exists!" << std::endl;
                return false;
            }

            // Returns true if an error has occurred on the associated stream.
            if (fs.fail())
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << "An error has occurred on the associated stream from opening '" << fileName << "' ." << std::endl;
                return false;
            }

            return true;
        }
        else
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "'" << fileName << "' does not exists!" << std::endl;
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

            return;
        }
        if (fs.is_open())
        {
            fs.close();
        }
    }

    std::fstream &getFstream()
    {
        return io::fs;
    }

    FILE *getFile()
    {
        return f;
    }

    char *getLine()
    {
        return line;
    }

    char **getLineArg()
    {
        return lineArg;
    }

  private:
    FILE *f;
    std::fstream fs;

    char *line;
    char **lineArg;

    // void fprint_matrix_1d(FILE *fp, char *title, double *v, int n)
    // {
    //     int i;

    //     if (fp == stdout)
    //         fprintf(fp, "\n%s =\n\n", title);
    //     for (i = 0; i < n; i++)
    //     {
    //         fprintf(fp, "%12.4lf ", v[i]);
    //     }
    //     fprintf(fp, "\n");
    // }

    // void fprint_matrix_2d(FILE *fp, char *title, double **v, int n1, int n2)
    // {
    //     int i, j;

    //     if (fp == stdout)
    //         fprintf(fp, "\n%s =\n\n", title);
    //     for (i = 0; i < n1; i++)
    //     {
    //         for (j = 0; j < n2; j++)
    //         {
    //             fprintf(fp, "   %20.15lf", v[i][j]);
    //         }
    //         fprintf(fp, "\n");
    //     }
    //     fprintf(fp, "\n");
    // }
};

#endif