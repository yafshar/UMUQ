#ifndef UMHBM_TMCMC_DATATYPE_H
#define UMHBM_TMCMC_DATATYPE_H

#include <iostream>
//malloc, calloc, qsort
#include <cstdlib>
//fopen, fgets, sscanf, sprintf
#include <cstdio>
//strlen, strstr
#include <cstring>

/*! \file TMCMC_datatype.hpp
*   \brief Data types and helper structures.
*
* \param Nth
* \param MaxStages
* \param PopSize
* \param lowerbound
* \param upperbound
* \param compositeprior_distr
* \param prior_mu
* \param prior_sigma
* \param auxil_size
* \param auxil_data
* \param MinChainLength
* \param MaxChainLength
* \param lb
* \param ub
* \param TolCOV
* \param bbeta
* \param seed
* \param options
* \param sampling_type
* \param prior_type
* \param prior_count
* \param iplot
* \param icdump              dump current dataset of accepted points
* \param ifdump              dump complete dataset of points
* \param Num
* \param LastNum
* \param use_proposal_cma
* \param init_mean
* \param local_cov
* \param use_local_cov
* \param local_scale
*/
struct data_t
{
	int Nth;
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

		//! constructor
		/*!
    	*  \brief constructor for the default variables
		*    
    	*/
		optim_options() : MaxIter(100),
						  Tol(1e-6),
						  Display(0),
						  Step(1e-5){};
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
	//! constructor
	/*!
	*  \brief constructor for the default variables
	*    
	*/
	data_t() : Nth(0),
			   MaxStages(0),
			   PopSize(0),
			   lowerbound(NULL),
			   upperbound(NULL),
			   compositeprior_distr(NULL),
			   prior_mu(NULL),
			   prior_sigma(NULL),
			   auxil_size(0),
			   auxil_data(NULL),
			   MinChainLength(0),
			   MaxChainLength(1e6),
			   lb(0), /* Default LB, same for all */
			   ub(0),
			   TolCOV(0),
			   bbeta(0.2),
			   seed(280675),
			   options(),
			   prior_type(1),
			   prior_count(0),
			   iplot(0),
			   icdump(1),
			   ifdump(0),
			   Num(NULL),
			   LastNum(0),
			   use_proposal_cma(0),
			   init_mean(NULL),
			   local_cov(NULL),
			   use_local_cov(0),
			   local_scale(0){};

	//! constructor
	/*!
    *  \brief constructor for the default input variables
	*    
    */
	data_t(int probdim, int maxgens, int datanum) : Nth(probdim),
													MaxStages(maxgens),
													PopSize(datanum),
													lowerbound(NULL),
													upperbound(NULL),
													compositeprior_distr(NULL),
													prior_mu(NULL),
													prior_sigma(NULL),
													auxil_size(0),
													auxil_data(NULL),
													MinChainLength(0),
													MaxChainLength(1e6),
													lb(-6), /* Default LB, same for all */
													ub(6),
													TolCOV(1.0),
													bbeta(0.2),
													seed(280675),
													options(),
													prior_type(1),
													prior_count(0),
													iplot(0),
													icdump(1),
													ifdump(0),
													Num(NULL),
													LastNum(datanum),
													use_proposal_cma(0),
													init_mean(NULL),
													local_cov(NULL),
													use_local_cov(0),
													local_scale(0)
	{
		lowerbound = new double[Nth](0);
		upperbound = new double[Nth](0); 
		prior_mu = (double *)calloc(Nth, Nth * sizeof(double));
		prior_sigma = (double *)calloc(Nth * Nth, Nth * Nth * sizeof(double));
		for (int i = 0; i < Nth; i++)
		{
			for (int j = 0; j < Nth; j++)
			{
				if (i == j)
				{
					prior_sigma[i * Nth + j] = 1.0;
				}
			}
		}
		Num = (int *)malloc(MaxStages * sizeof(int));
		for (int i = 0; i < Nth; i++)
		{
			Num[i] = PopSize;
		}
	};

	/*!
    *  \brief constructor for the default input variables
	*    
    */
	void read(const char *fname);
	void read();
};

// read the tmcmc.par file for setting the input variables
void data_t::read(const char *fname)
{
	FILE *f = fopen(fname, "r");

	char line[256];

	int line_no = 0;
	while (fgets(line, 256, f) != NULL)
	{
		line_no++;
		if ((line[0] == '#') || (strlen(line) == 0))
		{
			//std::cout << "ignoring line " << line_no << std::endl;
			continue;
		}

		if (strstr(line, "Nth"))
		{
			sscanf(line, "%*s %d", &Nth);
		}
		else if (strstr(line, "MaxStages"))
		{
			sscanf(line, "%*s %d", &MaxStages);
		}
		else if (strstr(line, "PopSize"))
		{
			sscanf(line, "%*s %d", &PopSize);
		}
		else if (strstr(line, "TolCOV"))
		{
			sscanf(line, "%*s %lf", &TolCOV);
		}
		else if (strstr(line, "bbeta"))
		{
			sscanf(line, "%*s %lf", &bbeta);
		}
		else if (strstr(line, "seed"))
		{
			sscanf(line, "%*s %ld", &seed);
		}
		else if (strstr(line, "opt.MaxIter"))
		{
			sscanf(line, "%*s %d", &options.MaxIter);
		}
		else if (strstr(line, "opt.Tol"))
		{
			sscanf(line, "%*s %lf", &options.Tol);
		}
		else if (strstr(line, "opt.Display"))
		{
			sscanf(line, "%*s %d", &options.Display);
		}
		else if (strstr(line, "opt.Step"))
		{
			sscanf(line, "%*s %lf", &options.Step);
			printf("setting step = %f\n", options.Step);
		}
		else if (strstr(line, "prior_type"))
		{
			sscanf(line, "%*s %d", &prior_type);
		}
		else if (strstr(line, "prior_count"))
		{
			sscanf(line, "%*s %d", &prior_count);
		}
		else if (strstr(line, "iplot"))
		{
			sscanf(line, "%*s %d", &iplot);
		}
		else if (strstr(line, "icdump"))
		{
			sscanf(line, "%*s %d", &icdump);
		}
		else if (strstr(line, "ifdump"))
		{
			sscanf(line, "%*s %d", &ifdump);
		}
		else if (strstr(line, "Bdef"))
		{
			sscanf(line, "%*s %lf %lf", &lb, &ub);
		}
		else if (strstr(line, "MinChainLength"))
		{
			sscanf(line, "%*s %d", &MinChainLength);
		}
		else if (strstr(line, "MaxChainLength"))
		{
			sscanf(line, "%*s %d", &MaxChainLength);
		}
		else if (strstr(line, "use_local_cov"))
		{
			sscanf(line, "%*s %d", &use_local_cov);
		}
	}

	if (lowerbound != NULL)
	{
		delete lowerbound;
	}
	if (upperbound != NULL)
	{
		delete upperbound;
	}

	lowerbound = (double *)malloc(Nth * sizeof(double));
	upperbound = (double *)malloc(Nth * sizeof(double));

	int found;
	for (int i = 0; i < Nth; i++)
	{
		rewind(f);
		found = 0;
		while (fgets(line, 256, f) != NULL)
		{
			if ((line[0] == '#') || (strlen(line) == 0))
			{
				continue;
			}

			char bound[8];
			sprintf(bound, "B%d", i);
			if (strstr(line, bound) != NULL)
			{
				sscanf(line, "%*s %lf %lf", &lowerbound[i], &upperbound[i]);
				found = 1;
				break;
			}
		}
		if (!found)
		{
			lowerbound[i] = lb; /* Bdef value or Default LB */
			upperbound[i] = ub; /* Bdef value of Default UB */
		}
	}

	if (prior_type == 1) /* gaussian */
	{
		/* new, parse prior_mu */
		rewind(f);
		line_no = 0;

		if (prior_mu != NULL)
		{
			delete prior_mu;
		}

		free(prior_mu);
		data.prior_mu = (double *)malloc(data.Nth * sizeof(double));

		found = 0;
		while (fgets(line, 256, f) != NULL)
		{
			line_no++;
			if ((line[0] == '#') || (strlen(line) == 0))
				continue;

			if (strstr(line, "prior_mu") != NULL)
			{
				char *tok = strtok(line, " ;,\t");
				if (tok == NULL)
					break;
				int i = 0;
				tok = strtok(NULL, " ;,\t");
				while (tok != NULL)
				{
					data.prior_mu[i] = atof(tok);
					i++;
					tok = strtok(NULL, " ;,\t");
				}
				found = 1;
				break;
			}
		}

		if (!found)
		{
			for (i = 0; i < data.Nth; i++)
			{
				data.prior_mu[i] = 0.0; /* Mudef value of Default Mean */
			}
		}

		/* new, parse prior_sigma */
		rewind(f);
		line_no = 0;

		free(data.prior_sigma);
		data.prior_sigma = (double *)malloc(data.Nth * data.Nth * sizeof(double));

		found = 0;
		while (fgets(line, 256, f) != NULL)
		{
			line_no++;
			if ((line[0] == '#') || (strlen(line) == 0))
				continue;

			if (strstr(line, "prior_sigma") != NULL)
			{
				char *tok = strtok(line, " ;,\t");
				if (tok == NULL)
					break;
				int i = 0;
				tok = strtok(NULL, " ;,\t");
				while (tok != NULL)
				{
					data.prior_sigma[i] = atof(tok);
					i++;
					tok = strtok(NULL, " ;,\t");
				}
				found = 1;
				break;
			}
		}

		if (!found)
		{
			for (i = 0; i < data.Nth; i++)
			{
				int j;
				for (j = 0; j < data.Nth; j++)
				{
					if (i == j)
						data.prior_sigma[i * data.Nth + j] = 1.0; /* Sigmadef value of Default Sigma */
					else
						data.prior_sigma[i * data.Nth + j] = 0.0;
				}
			}
		}
	}

	if (data.prior_type == 3) /* composite */
	{
		rewind(f);
		line_no = 0;

		data.compositeprior_distr = (double *)malloc(data.Nth * sizeof(double));

		free(data.prior_mu);
		free(data.prior_sigma);
		data.prior_mu = (double *)malloc(data.Nth * sizeof(double));
		data.prior_sigma = (double *)malloc(data.Nth * data.Nth * sizeof(double));

		for (i = 0; i < data.Nth; i++)
		{
			found = 0;
			while (fgets(line, 256, f) != NULL)
			{
				line_no++;

				if ((line[0] == '#') || (strlen(line) == 0))
					continue;

				char bound[8];
				sprintf(bound, "C%d", i);
				if (strstr(line, bound) != NULL)
				{
					sscanf(line, "%*s %lf %lf %lf", &data.compositeprior_distr[i],
						   &data.lowerbound[i], &data.upperbound[i]);
					found = 1;
					break;
				}
			}
			if (!found)
			{
				data.lowerbound[i] = data.lb; /* Bdef value or Default LB */
				data.upperbound[i] = data.ub; /* Bdef value of Default UB */
				data.compositeprior_distr[i] = 0;
			}
			rewind(f);
			line_no = 0;
		}
	}

	/* new, parse auxil_size and auxil_data */
	rewind(f);
	line_no = 0;

	found = 0;
	while (fgets(line, 256, f) != NULL)
	{
		line_no++;
		if ((line[0] == '#') || (strlen(line) == 0))
			continue;

		if (strstr(line, "auxil_size") != NULL)
		{
			sscanf(line, "%*s %d", &data.auxil_size);
			found = 1;
			break;
		}
	}

	if (data.auxil_size > 0)
	{
		rewind(f);
		line_no = 0;

		data.auxil_data = (double *)malloc(data.auxil_size * sizeof(double));

		found = 0;
		while (fgets(line, 256, f) != NULL)
		{
			line_no++;
			if ((line[0] == '#') || (strlen(line) == 0))
				continue;

			if (strstr(line, "auxil_data") != NULL)
			{
				char *tok = strtok(line, " ;,\t");
				if (tok == NULL)
					break;
				int i = 0;
				tok = strtok(NULL, " ;,\t");
				while (tok != NULL)
				{
					data.auxil_data[i] = atof(tok);
					i++;
					tok = strtok(NULL, " ;,\t");
				}
				found = 1;
				break;
			}
		}
	}

	fclose(f);

#if 0
    print_matrix((char *)"prior_mu", data.prior_mu, data.Nth);
    print_matrix((char *)"prior_sigma", data.prior_sigma, data.Nth*data.Nth);
    print_matrix((char *)"auxil_data", data.auxil_data, data.auxil_size);
#endif

	free(data.Num);
	data.Num = (int *)malloc(data.MaxStages * sizeof(int));
	for (i = 0; i < data.MaxStages; i++)
	{
		data.Num[i] = data.PopSize;
	}
	data.LastNum = data.PopSize;

	double *LCmem = (double *)calloc(1, data.PopSize * data.Nth * data.Nth * sizeof(double));
	data.local_cov = (double **)malloc(data.PopSize * sizeof(double *));
	int pos;
	for (pos = 0; pos < data.PopSize; ++pos)
	{
		data.local_cov[pos] = LCmem + pos * data.Nth * data.Nth;
		for (i = 0; i < data.Nth; ++i)
			data.local_cov[pos][i * data.Nth + i] = 1;
	}
}

/*!
*  \brief basic structure
*    
*  \param Parray     double array for points in space
*  \param ndimParray an integer argument shows the size of Parray
*  \param Garray     double array
*  \param ndimGarray an integer argument shows the size of Garray
*  \param Fvalue     double argument for the function value
*  \param surrogate  an integer argument shows the surrogate model
*  \param nsel       an integer argument for selection of leaders only
*/
struct basic
{
	double *Parray;
	int ndimParray;
	double *Garray;
	int ndimGarray;
	double Fvalue;
	int surrogate;
	int nsel;
	/*!
    *  \brief constructor for the default variables
	*    
    */
	basic() : Parray(NULL),
			  ndimParray(0),
			  Garray(NULL),
			  ndimGarray(0),
			  Fvalue(0),
			  surrogate(0),
			  nsel(0){};
};

/*!
*  \brief current generation structure
*    
* \param queue an integer argument for submission of leaders only
* \param error double argument for measuring error
*/
struct cgdbp : basic
{

	int queue;
	double error;
	/*!
    *  \brief constructor for the default variables
	*    
    */
	cgdbp() : queue(0),
			  error(0){};
};

/*!
*  \brief database generation structure
*    
*/
struct dbp : basic
{
};

/*!
*  \brief database generation structure
*    
*/
struct resdbp : basic
{
};

/*!
*  \brief run info structure
*    
* \param Gen    
* \param CoefVar        The coefficient of variation of the plausibility weights 
* \param p              cluster-wide
* \param currentuniques   
* \param logselection   
* \param acceptance  
* \param SS             cluster-wide
* \param meantheta  
*/
struct runinfo
{
	int Gen;
	double *CoefVar;					 /*[MAXGENS];*/
	double *p;							 /*[MAXGENS];*/
	int *currentuniques;				 /*[MAXGENS];*/
	double *logselection;				 /*[MAXGENS];*/
	double *acceptance;					 /*[MAXGENS];*/
	double **SS; /*[PROBDIM][PROBDIM];*/ //
	double **meantheta;					 /*[MAXGENS][PROBDIM];*/
};

/*!
*  \brief structure for sorting Fvalue for entires of database structure
* 
* \param idx      an intger argument for indexing
* \param nsel     an integer argument for selection of leaders only
* \param Fvalue   a double argument for function value
*/
struct sort_t
{
	int idx;
	int nsel;
	double Fvalue;
};

/*!
*  \brief database structure
*
* \param entry
* \param entries an integer argument shows the size of entry
* \param m A mutex object
*/
template <class T>
struct database
{
	T *entry;
	int entries;
	pthread_mutex_t m;

	/*!
    *  \brief constructor for the database structure
	*    
    */
	database() : entry(NULL),
				 entries(0)
	{
		pthread_mutex_init(&m, NULL);
	};

	/*!
	* /brief Init function taking two arguments and initialize the structure.
	*
    *  \param nsize1 an integer argument.
    *  \param nsize2 an integer argument.
    */
	void init(int nsize1, int nsize2);
	void init(int nsize1);

	/*!
	* /brief A member updating the database
	*
    *  \param Parray     a double array of points.
    *  \param ndimParray an integer argument, shows the size of Parray
	*  \param Fvalue     a double value 
	*  \param Garray     a double array
	*  \param ndimGarray an integer argument, shows the size of Garray
	*  \param surrogate  an integer argument for the surrogate model
    */
	void update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray, int surrogate);
	void update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray);

	/*!
	* /brief function for sorting elemnts of an array for database elements.
	*
	*  Sorts the entries elements of the array pointed to by list, each 
	*  element size bytes long, using the compar function to determine the order.
    */
	void sort(sort_t *list);
};

template <class T>
void database<T>::init(int nsize1, int nsize2)
{
	if (entry == NULL)
	{
		entry = (T *)calloc(1, nsize1 * nsize2 * sizeof(T));
	}
}

template <class T>
void database<T>::init(int nsize1)
{
	if (entry == NULL)
	{
		entry = (T *)calloc(1, nsize1 * sizeof(T));
	}
}

template <class T>
void database<T>::update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray, int surrogate)
{
	int pos;

	pthread_mutex_lock(&m);
	pos = entries;
	entries++;
	pthread_mutex_unlock(&m);

	if (ndimParray > entry[pos].ndimParray)
	{
		entry[pos].Parray = (double *)realloc(entry[pos].Parray, ndimParray * sizeof(double));
	}
	entry[pos].ndimParray = ndimParray;

	for (int i = 0; i < ndimParray; i++)
	{
		entry[pos].Parray[i] = Parray[i];
	}

	entry[pos].Fvalue = Fvalue;

	if (ndimGarray > entry[pos].ndimGarray)
	{
		entry[pos].Garray = (double *)realloc(entry[pos].Garray, ndimGarray * sizeof(double));
	}

	entry[pos].ndimGarray = ndimGarray;

	for (int i = 0; i < ndimGarray; i++)
	{
		entry[pos].Garray[i] = Garray[i];
	}

	entry[pos].surrogate = surrogate;
};

template <class T>
void database<T>::update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray)
{
	int pos;

	pthread_mutex_lock(&m);
	pos = entries;
	entries++;
	pthread_mutex_unlock(&m);

	if (ndimParray > entry[pos].ndimParray)
	{
		entry[pos].Parray = (double *)realloc(entry[pos].Parray, ndimParray * sizeof(double));
	}
	entry[pos].ndimParray = ndimParray;

	for (int i = 0; i < ndimParray; i++)
	{
		entry[pos].Parray[i] = Parray[i];
	}

	entry[pos].Fvalue = Fvalue;

	if (ndimGarray > entry[pos].ndimGarray)
	{
		entry[pos].Garray = (double *)realloc(entry[pos].Garray, ndimGarray * sizeof(double));
	}

	entry[pos].ndimGarray = ndimGarray;

	for (int i = 0; i < ndimGarray; i++)
	{
		entry[pos].Garray[i] = Garray[i];
	}
};

/*!
*  \brief Pointer to a function that compares two elements.
*   
*  This function is called repeatedly by qsort to compare two elements.
*/
int compar_desc(const void *p1, const void *p2)
{
	sort_t *s1 = (sort_t *)p1;
	sort_t *s2 = (sort_t *)p2;

	/* -: ascending order, +: descending order */
	return (s2->nsel - s1->nsel);
}

template <class T>
void database<T>::sort(sort_t *list)
{
	if (list == NULL)
	{
		list = (sort_t *)malloc(entries * sizeof(sort_t));
	}
	for (int i = 0; i < entries; i++)
	{
		list[i].idx = i;
		list[i].nsel = entry[i].nsel;
		list[i].Fvalue = entry[i].Fvalue;
	}
	qsort(list, entries, sizeof(sort_t), compar_desc);
}

struct cgdb : database<cgdbp>
{
};

struct db : database<dbp>
{
};

struct resdb : database<resdbp>
{
};

#endif