#ifndef UMHBM_UTILITY_H
#define UMHBM_UTILITY_H

#define _XOPEN_SOURCE 700
#define _BSD_SOURCE 1

#include <iostream>
//fopen, remove, perror, sprintf
#include <stdio.h>
//stat, open, fstat
#include <sys/stat.h>
//strlen, strstr, strtok
#include <cstring>
//fork, execvp, chdir, fstat
#include <unistd.h>
//waitpid, fstat, opendir
#include <sys/types.h>
//waitpid
#include <sys/wait.h>
//nftw
#include <ftw.h>
//open
#include <fcntl.h>
//perror
#include <errno.h>
//opendir, readdir
#include <dirent.h>

#define LINESIZE 256

/*! \class utility
*   \brief utility is a class which includes some helper functionality.
*	
* 	utility class contains functionality for parsing the line, exectuing command
*   delete a name from a filesystem or unlink the files, copy file from a path to 
*   other path
*/

class utility
{
  public:
    FILE *f;

    char *line;
    char **lineArg;

    utility() : f(NULL), line(NULL), lineArg(NULL){};

    ~utility()
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

    // /*!
    //  *  \brief Execute a command getting the std::cout
    //  *
    //  */
    // static inline std::string exec(const char *argv)
    // {
    //     std::array<char, 256> buffer;
    //     std::string result;
    //     std::shared_ptr<FILE> pipe(popen(argv, "r"), pclose);
    //     if (!pipe)
    //     {
    //         throw std::runtime_error("popen() failed!");
    //     }
    //     while (!feof(pipe.get()))
    //     {
    //         if (fgets(buffer.data(), 256, pipe.get()) != NULL)
    //         {
    //             result += buffer.data();
    //         }
    //     }
    //     return result;
    // }

    int execute_cmd(int me, char **argv, const char *dir)
    {
        pid_t pid;

        // fork a child process
        if ((pid = fork()) < 0)
        {
            std::cout << "spanwer(" << me << "): forking child process failed!!!!" << std::endl;
            return 1;
        }
        else if (pid == 0) // child process
        {
            if (dir != NULL)
            {
                // move to the specified directory
                if (chdir(dir) < 0)
                {
                    std::cout << "Failed to change PATH!!!!" << std::endl;
                    std::cout << "Permission is denied for one of the components in the path to : " << dir << std::endl;
                    return 1;
                }
            }

            if (execvp(*argv, argv) < 0)
            {
                std::cout << "Error occured while attempting to execute the command!!!!" << std::endl;
                return 1;
            }
        }

        // wait for process to change state
        waitpid(pid, NULL, 0);

        return 0;
    }

    static int unlink_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
    {
        // deletes a name from the file system. It calls unlink for files, and rmdir for directories.
        int rv = remove(fpath);

        // produces a message on the standard error output, describing the last error encountered during
        // a call to a system or library function
        if (rv)
        {
            perror(fpath);
        }

        return rv;
    }

    int rmrf(char *path)
    {
        // file tree walk
        // FTW_DEPTH: do a post-order traversal
        // FTW_PHYS: do not follow symbolic links
        return nftw(path, unlink_cb, 64, FTW_DEPTH | FTW_PHYS);
    }

    int cp(const char *from, const char *to)
    {
        int fd_to, fd_from;
        char buf[4096];
        ssize_t nread;
        int saved_errno;
        struct stat sb;

        // O_RDONLY: Open for reading only
        fd_from = open(from, O_RDONLY);
        if (fd_from < 0)
        {
            return -1;
        }

        // get file status
        fstat(fd_from, &sb);
        if (S_ISDIR(sb.st_mode))
        {
            fd_to = -1;
            goto out_error;
        }

        fd_to = open(to, O_WRONLY | O_CREAT | O_EXCL, sb.st_mode);
        if (fd_to < 0)
        {
            goto out_error;
        }

        while (nread = read(fd_from, buf, sizeof buf), nread > 0)
        {
            char *out_ptr = buf;
            ssize_t nwritten;

            do
            {
                nwritten = write(fd_to, out_ptr, nread);
                if (nwritten >= 0)
                {
                    nread -= nwritten;
                    out_ptr += nwritten;
                }
                else if (errno != EINTR)
                {
                    goto out_error;
                }
            } while (nread > 0);
        }

        if (nread == 0)
        {
            if (close(fd_to) < 0)
            {
                fd_to = -1;
                goto out_error;
            }
            else
            { // peh: added due to some issues on monte rosa
                fsync(fd_to);
            }
            close(fd_from);

            /* Success! */
            return 0;
        }

    out_error:
        saved_errno = errno;

        close(fd_from);
        if (fd_to >= 0)
        {
            close(fd_to);
        }

        errno = saved_errno;
        return -1;
    }

    int copy_from_dir(char *name)
    {
        DIR *dir;
        struct dirent *ent;

        // open a directory
        dir = opendir(name);
        if (dir == NULL)
        {
            // could not open directory
            perror("The following error occurred");
            return 1;
        }
        else
        {
            // read a directory
            /* print all the files and directories within directory */
            while ((ent = readdir(dir)) != NULL)
            {
                char source[256], dest[256];

                sprintf(source, "%s/%s", name, ent->d_name);
                sprintf(dest, "./%s", ent->d_name);
                cp(source, dest);
            }
            closedir(dir);
        }
        return 0;
    }

    int copy_file(const char *dirname, const char *fileName)
    {
        DIR *dir;

        // open a directory
        dir = opendir(dirname);
        if (dir == NULL)
        {
            // could not open directory
            perror("The following error occurred");
            return 1;
        }
        else
        {
            char source[256], dest[256];

            sprintf(source, "%s/%s", dirname, fileName);
            sprintf(dest, "./%s", fileName);

            cp(source, dest);

            closedir(dir);
        }

        return 0;
    }
};

#endif