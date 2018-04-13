#ifndef UMHBM_FNCOUNTER_H
#define UMHBM_FNCOUNTER_H

#include "../../external/torc/include/torc.h"

/*! \class fncounter
*   \brief Function call counters
*	
*/
struct fncounter
{
	static pthread_mutex_t fm;

	static int nflcl;
	static int nfglb;
	static int nftot;

	void increment()
	{
		//lock and unlock a mutex
		pthread_mutex_lock(&fm);
		nflcl++;
		pthread_mutex_unlock(&fm);
	}

	void reset_task() { nflcl = 0; }

	void reset()
	{
		for (int i = 0; i < torc_num_nodes(); i++)
		{
			torc_create_ex(i * torc_i_num_workers(), 1, (void *)reset_task, 0);
		}
		torc_waitall();
	}

	void get_task(int *x) { *x = nflcl; }

	int get()
	{
		//TODO correct the number of nodes
		int c[1024]; /* MAX_NODES*/

		for (int i = 0; i < torc_num_nodes(); i++)
		{
			torc_create_ex(i * torc_i_num_workers(), 1, (void *)get_task, 1,
						   1, MPI_INT, CALL_BY_RES,
						   &c[i]);
		}
		torc_waitall();

		nfglb = std::accumulate(c, c + torc_num_nodes(), 0);

		std::cout << "global number of function counts:" << nfglb << std::endl;

		nftot += nfglb;

		return nfglb;
	}
};

pthread_mutex_t fncounter::fm = PTHREAD_MUTEX_INITIALIZER;

int fncounter::nflcl = 0;
int fncounter::nfglb = 0;
int fncounter::nftot = 0;

#endif
