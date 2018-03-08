#ifndef UMHBM_HELPER_H
#define UMHBM_HELPER_H

#define _XOPEN_SOURCE 700
#define _BSD_SOURCE 1

#include <iostream>
//waitpid, fstat, opendir
#include <sys/types.h>
//waitpid
#include <sys/wait.h>
//fork, execvp, chdir, fstat
#include <unistd.h>
//remove, perror, sprintf
#include <stdio.h>
//nftw
#include <ftw.h>
//open,fstat
#include <sys/stat.h>
//open
#include <fcntl.h>
//perror
#include <errno.h>
//opendir, readdir
#include <dirent.h>

/*! \class spawner
*   \brief spawner is a class which includes some helper functionality.
*	
* 	spawner class contains functionality for parsing the line, exectuing command
*   delete a name from a filesystem or unlink the files, copy file from a path to 
*   other path
*/

class spawner
{
  public:
	void parse(char *line, char **argv)
	{
		/* if not the end of line ....... */
		while (*line != '\0')
		{
			while (*line == ' ' || *line == '\t' || *line == '\n')
			{
				/* replace white spaces with 0 */
				*line++ = '\0';
			}

			/* save the argument position */
			*argv++ = line;

			/* skip the argument until ...*/
			while (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n')
			{
				line++;
			}
		}

		/* mark the end of argument list */
		*argv = NULL;
	}

	int execute_cmd(int me, char **argv, char *dir)
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
				chdir(dir);
			}

			execvp(*argv, argv);
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
			perror(fpath);

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
			return -1;

		// get file status
		fstat(fd_from, &sb);
		if (S_ISDIR(sb.st_mode))
		{
			fd_to = -1;
			goto out_error;
		}

		fd_to = open(to, O_WRONLY | O_CREAT | O_EXCL, sb.st_mode);
		if (fd_to < 0)
			goto out_error;

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
			close(fd_to);

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

	int copy_file(const char *dirname, const char *filename)
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

			sprintf(source, "%s/%s", dirname, filename);
			sprintf(dest, "./%s", filename);

			cp(source, dest);

			closedir(dir);
		}

		return 0;
	}
};

#endif