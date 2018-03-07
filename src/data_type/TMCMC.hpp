#ifndef UMHBM_TMCMC_H
#define UMHBM_TMCMC_H

// #include <math.h>
// #include <mpi.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <sys/types.h>
// #include <torc.h>
// #include <unistd.h>

// #include "gsl_headers.h"

// #define EXPERIMENTAL_RESULTS  0
template <unsigned int probdim, unsigned int maxgens, unsigned int datanum>
class TMCMC
{
	/*** HELPER STRUCTS ***/
	struct data_s
	{
		int Nth;	   /* = PROBDIM*/
		int MaxStages; /* = MAXGENS*/
		int PopSize;   /* = DATANUM*/

		double *lowerbound; /*[PROBDIM];*/
		double *upperbound; /*[PROBDIM];*/

		double *compositeprior_distr; /*[PROBDIM]*/

		double *prior_mu;
		double *prior_sigma;

		int auxil_size;
		double *auxil_data;

		int MinChainLength;
		int MaxChainLength;

		double lb; /*generic lower bound*/
		double ub; /*generic upper bound*/

		double TolCOV; /*a prescribed tolerance*/
		double bbeta;
		long seed;

		struct optim_options
		{
			int MaxIter;
			double Tol;
			int Display;
			double Step;
		} options;

		int sampling_type; /* 0: uniform, 1: gaussian, 2: file */
		int prior_type;	/* 0: lognormal, 1: gaussian */

		/* prior information needed for hiegherarchical analysis */
		/* this number = prior + number of datasets = 1 + N_IND */
		/* if it is 0, we only do the TMCMC */
		int prior_count;

		int iplot;
		int icdump;
		int ifdump;

		int *Num; /*[MAXGENS];*/
		int LastNum;

		int use_proposal_cma;
		double **init_mean; /* [DATANUM][PROBDIM] */

		double **local_cov; /* [DATANUM][PROBDIM*PROBDIM] */
		int use_local_cov;
		double local_scale;

		data_s(){
			Nth = (int) probdim;
			MaxStages = (int) MaxStages;
			PopSize = (int) datanum;
		};
	};

	struct runinfo_s
	{
		int Gen;
		double *CoefVar; /*[MAXGENS];*/		 // The coefficient of variation of the plausibility weights
		double *p; /*[MAXGENS];*/			 // cluster-wide
		int *currentuniques;				 /*[MAXGENS];*/
		double *logselection;				 /*[MAXGENS];*/
		double *acceptance;					 /*[MAXGENS];*/
		double **SS; /*[PROBDIM][PROBDIM];*/ // cluster-wide
		double **meantheta;					 /*[MAXGENS][PROBDIM];*/
	};

	struct sort_s
	{
		int idx;
		int nsel;
		double F;
	};

	/*** DATABASES ***/
	struct cgdbp_s
	{
		double *point; /*[PROBDIM];*/
		double F;
		double *prior;
		int counter; /* I used it for prior */

		int nsel;  /* for selection of leaders only*/
		int queue; /* for submission of leaders only*/
				   /* NN */
		int surrogate;
		double error;

#if defined(_TMCMC_SN_)
		int valid;
		double *grad;
		double *hes;
#endif
	};

	struct cgdb_s
	{
		cgdbp_t *entry; /*[MAX_DB_ENTRIES];*/
		int entries;
		pthread_mutex_t m;
	};

	struct dbp_s
	{
		double *point; /*[PROBDIM];*/
		double F;
		int nG;
		double G[64]; /* maxG*/
		int surrogate;
	};

	struct db_s
	{
		dbp_t *entry; /*[MAX_DB_ENTRIES];*/ /* */
		int entries;
		pthread_mutex_t m;
	};

	struct resdbp_s
	{
		double *point; /*[EXPERIMENTAL_RESULTS+1]; // +1 for the result (F)*/
		double F;
		int counter; /* not used (?)*/
		int nsel;	/* for selection of leaders only*/
	};

	struct resdb_s
	{
		resdbp_t *entry; /*[MAX_DB_ENTRIES];*/
		int entries;
		pthread_mutex_t m;
	};

  public:
	typedef struct data_s data_t;
	typedef struct runinfo_s runinfo_t;
	typedef struct sort_s sort_t;
	typedef struct cgdbp_s cgdbp_t;
	typedef struct cgdb_s cgdb_t;
	typedef struct dbp_s dbp_t;
	typedef struct db_s db_t;
	typedef struct resdbp_s resdbp_t;
	typedef struct resdb_s resdb_t;

	/*** DATABASE INSTANCES ***/
	data_t data;
	runinfo_t runinfo;
	cgdb_t curgen_db;
	db_t full_db;
	resdb_t curres_db;
};

/*** END HELPER STRUCTS ***/

#endif
