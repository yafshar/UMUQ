#ifndef UMHBM_TMCMC_DATATYPE_H
#define UMHBM_TMCMC_DATATYPE_H

#include <iostream>
#include <iomanip>
#include <system_error>

//malloc, calloc, qsort, atof
#include <cstdlib>
//fopen, fgets, sscanf, sprintf
#include <cstdio>
//strlen, strstr, strtok
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
    int prior_type;    /* 0: lognormal, 1: gaussian */

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
               TolCOV(1.0),
               bbeta(0.2),
               seed(280675),
               options(),
               prior_type(0),
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
                                                    prior_type(0),
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
        lowerbound = new double[Nth];
        upperbound = new double[Nth];
        prior_mu = new double[Nth];
        int e = Nth;
        while (e--)
        {
            lowerbound[e] = 0;
            upperbound[e] = 0;
            prior_mu[e] = 0;
        }

        prior_sigma = new double[Nth * Nth];
        for (int i = 0, k = 0; i < Nth; i++)
        {
            for (int j = 0; j < Nth; j++, k++)
            {
                if (i == j)
                {
                    prior_sigma[k] = 1.0;
                }
                else
                {
                    prior_sigma[k] = 0.0;
                }
            }
        }

        Num = new int[MaxStages];
        e = MaxStages;
        while (e--)
            Num[e] = PopSize;

        local_cov = new double *[PopSize];
        for (int i = 0; i < PopSize; i++)
        {
            local_cov[i] = new double[Nth * Nth];
            for (int j = 0, l = 0; j < Nth; j++)
            {
                for (int k = 0; k < Nth; k++, l++)
                {
                    if (j == k)
                    {
                        local_cov[i][l] = 1.0;
                    }
                    else
                    {
                        local_cov[i][l] = 0.0;
                    }
                }
            }
        }
    };

    /*!
    *  \brief constructor for the default input variables
    *    
    */
    void read(const char *fname);
    void read()
    {
        // read the tmcmc.par file for setting the input variables
        read("tmcmc.par");
    };

    //! destructor
    /*!
    *  \brief destructor 
    *    
    */
    ~data_t()
    {
        if (lowerbound != NULL)
        {
            delete[] lowerbound;
            lowerbound = NULL;
        }
        if (upperbound != NULL)
        {
            delete[] upperbound;
            upperbound = NULL;
        }
        if (compositeprior_distr != NULL)
        {
            delete[] compositeprior_distr;
            compositeprior_distr = NULL;
        }
        if (prior_mu != NULL)
        {
            delete[] prior_mu;
            prior_mu = NULL;
        }
        if (prior_sigma != NULL)
        {
            delete[] prior_sigma;
            prior_sigma = NULL;
        }
        if (auxil_data != NULL)
        {
            delete[] auxil_data;
            auxil_data = NULL;
        }
        if (Num != NULL)
        {
            delete[] Num;
            Num = NULL;
        }

        if (init_mean != NULL)
        {
            delete[] * init_mean;
            delete[] init_mean;
            init_mean = NULL;
        }

        if (local_cov != NULL)
        {
            delete[] * local_cov;
            delete[] local_cov;
            local_cov = NULL;
        }
    };

    void destroy();
};

// read the input file fname for setting the input variables
void data_t::read(const char *fname)
{
    FILE *f = fopen(fname, "r");

    char line[256];

    int probdim = Nth;
    int maxgens = MaxStages;
    int datanum = PopSize;
    bool linit;

    while (fgets(line, 256, f) != NULL)
    {
        if ((line[0] == '#') || (strlen(line) == 0))
        {
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

    linit = !(probdim == Nth && maxgens == MaxStages && datanum == PopSize && lowerbound != NULL);

    if (linit)
    {
        if (lowerbound != NULL)
        {
            delete[] lowerbound;
        }
        lowerbound = new double[Nth];

        if (upperbound != NULL)
        {
            delete[] upperbound;
        }
        upperbound = new double[Nth];
    }

    int i;
    int found;
    for (i = 0; i < Nth; i++)
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
        if (linit)
        {
            /* new, parse prior_mu */
            if (prior_mu != NULL)
            {
                delete[] prior_mu;
            }
            prior_mu = new double[Nth];
        }

        rewind(f);
        found = 0;
        while (fgets(line, 256, f) != NULL)
        {
            if ((line[0] == '#') || (strlen(line) == 0))
            {
                continue;
            }

            if (strstr(line, "prior_mu") != NULL)
            {
                char *tok = strtok(line, " ;,\t");
                if (tok == NULL)
                {
                    break;
                }

                i = 0;
                tok = strtok(NULL, " ;,\t");
                while (tok != NULL)
                {
                    prior_mu[i] = atof(tok);
                    i++;
                    tok = strtok(NULL, " ;,\t");
                }
                found = 1;
                break;
            }
        }

        if (!found)
        {
            i = Nth;
            while (i--)
                prior_mu[i] = 0.0;
        }

        if (linit)
        {
            /* new, parse prior_sigma */
            if (prior_sigma != NULL)
            {
                delete[] prior_sigma;
            }
            prior_sigma = new double[Nth * Nth];
        }

        rewind(f);
        found = 0;
        while (fgets(line, 256, f) != NULL)
        {
            if ((line[0] == '#') || (strlen(line) == 0))
            {
                continue;
            }

            if (strstr(line, "prior_sigma") != NULL)
            {
                char *tok = strtok(line, " ;,\t");
                if (tok == NULL)
                {
                    break;
                }

                i = 0;
                tok = strtok(NULL, " ;,\t");
                while (tok != NULL)
                {
                    prior_sigma[i] = atof(tok);
                    i++;
                    tok = strtok(NULL, " ;,\t");
                }
                found = 1;
                break;
            }
        }

        if (!found)
        {
            int j, k;
            for (i = 0, k = 0; i < Nth; i++)
            {
                for (j = 0; j < Nth; j++, k++)
                {
                    if (i == j)
                    {
                        prior_sigma[k] = 1.0;
                    }
                    else
                    {
                        prior_sigma[k] = 0.0;
                    }
                }
            }
        }
    }

    if (prior_type == 3) /* composite */
    {
        if (compositeprior_distr != NULL)
        {
            delete[] compositeprior_distr;
        }
        compositeprior_distr = new double[Nth];

        if (linit)
        {
            if (prior_mu != NULL)
            {
                delete[] prior_mu;
            }
            prior_mu = new double[Nth];

            if (prior_sigma != NULL)
            {
                delete[] prior_sigma;
            }
            prior_sigma = new double[Nth * Nth];
        }

        for (i = 0; i < Nth; i++)
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
                sprintf(bound, "C%d", i);
                if (strstr(line, bound) != NULL)
                {
                    sscanf(line, "%*s %lf %lf %lf", &compositeprior_distr[i], &lowerbound[i], &upperbound[i]);
                    found = 1;
                    break;
                }
            }
            if (!found)
            {
                lowerbound[i] = lb; /* Bdef value or Default LB */
                upperbound[i] = ub; /* Bdef value of Default UB */
                compositeprior_distr[i] = 0;
            }
        }
    }

    /* new, parse auxil_size and auxil_data */
    rewind(f);
    found = 0;
    while (fgets(line, 256, f) != NULL)
    {
        if ((line[0] == '#') || (strlen(line) == 0))
        {
            continue;
        }

        if (strstr(line, "auxil_size") != NULL)
        {
            sscanf(line, "%*s %d", &auxil_size);
            found = 1;
            break;
        }
    }

    if (auxil_size > 0)
    {
        if (auxil_data != NULL)
        {
            delete[] auxil_data;
        }
        auxil_data = new double[auxil_size];

        rewind(f);
        found = 0;
        while (fgets(line, 256, f) != NULL)
        {
            if ((line[0] == '#') || (strlen(line) == 0))
            {
                continue;
            }

            if (strstr(line, "auxil_data") != NULL)
            {
                char *tok = strtok(line, " ;,\t");
                if (tok == NULL)
                {
                    break;
                }

                i = 0;
                tok = strtok(NULL, " ;,\t");
                while (tok != NULL)
                {
                    auxil_data[i] = atof(tok);
                    i++;
                    tok = strtok(NULL, " ;,\t");
                }
                found = 1;
                break;
            }
        }
    }

    fclose(f);

    if (linit)
    {
        if (Num != NULL)
        {
            delete[] Num;
        }
        Num = new int[MaxStages];

        i = MaxStages;
        while (i--)
            Num[i] = PopSize;

        LastNum = PopSize;

        if (local_cov != NULL)
        {
            for (i = 0; i < PopSize; i++)
            {
                if (local_cov[i] != NULL)
                {
                    delete[] local_cov[i];
                }
            }
            delete[] local_cov;
        }

        local_cov = new double *[PopSize];
        for (i = 0; i < PopSize; i++)
        {
            local_cov[i] = new double[Nth * Nth];
            for (int j = 0, l = 0; j < Nth; j++)
            {
                for (int k = 0; k < Nth; k++, l++)
                {
                    if (j == k)
                    {
                        local_cov[i][l] = 1;
                    }
                    else
                    {
                        local_cov[i][l] = 0;
                    }
                }
            }
        }
    }
};

/*!
*  \brief destructor 
*    
*/
void data_t::destroy()
{
    if (lowerbound != NULL)
    {
        delete[] lowerbound;
        lowerbound = NULL;
    }

    if (upperbound != NULL)
    {
        delete[] upperbound;
        upperbound = NULL;
    }

    if (compositeprior_distr != NULL)
    {
        delete[] compositeprior_distr;
        compositeprior_distr = NULL;
    }

    if (prior_mu != NULL)
    {
        delete[] prior_mu;
        prior_mu = NULL;
    }

    if (prior_sigma != NULL)
    {
        delete[] prior_sigma;
        prior_sigma = NULL;
    }

    if (auxil_data != NULL)
    {
        delete[] auxil_data;
        auxil_data = NULL;
    }

    if (Num != NULL)
    {
        delete[] Num;
        Num = NULL;
    }

    if (init_mean != NULL)
    {
        delete[] * init_mean;
        delete[] init_mean;
        init_mean = NULL;
    }

    if (local_cov != NULL)
    {
        delete[] * local_cov;
        delete[] local_cov;
        local_cov = NULL;
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
struct cgdbp_t : basic
{

    int queue;
    double error;
    /*!
    *  \brief constructor for the default variables
    *    
    */
    cgdbp_t() : queue(0),
                error(0){};
};

/*!
*  \brief database generation structure
*    
*/
struct dbp_t : basic
{
};

/*!
*  \brief database generation structure
*    
*/
struct resdbp_t : basic
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
struct runinfo_t
{
    int Gen;
    double *CoefVar;                     /*[MAXGENS];*/
    double *p;                           /*[MAXGENS];*/
    int *currentuniques;                 /*[MAXGENS];*/
    double *logselection;                /*[MAXGENS];*/
    double *acceptance;                  /*[MAXGENS];*/
    double **SS; /*[PROBDIM][PROBDIM];*/ //
    double **meantheta;                  /*[MAXGENS][PROBDIM];*/
                                         /*!
    *  \brief constructor for the default variables
    *    
    */
    runinfo_t() : Gen(0),
                  CoefVar(NULL),
                  p(NULL),
                  currentuniques(NULL),
                  logselection(NULL),
                  acceptance(NULL),
                  SS(NULL),
                  meantheta(NULL){};
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
    void init(int nsize1);
    void init(int nsize1, int nsize2)
    {
        init(nsize1 * nsize2);
    };

    /*!
    * /brief A member updating the database
    *
    *  \param Parray     a double array of points.
    *  \param ndimParray an integer argument, shows the size of Parray
    *  \param Fvalue     a double value 
    *  \param Garray     a double array
    *  \param ndimGarray an integer argument, shows the size of Garray
    *  \param surrogate  an integer argument for the surrogate model (default 0, no surrogate)
    */
    void update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray, int surrogate);
    void update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray)
    {
        update(*Parray, ndimParray, Fvalue, *Garray, ndimGarray, 0);
    };

    /*!
    * /brief function for sorting elemnts of an array for database elements.
    *
    *  Sorts the entries elements of the array pointed to by list, each 
    *  element size bytes long, using the compar function to determine the order.
    */
    void sort(sort_t *list);

    /*!
    * /brief function for printing  the data
    *
    */
    virtual void print(){};

    /*!
    * /brief function for dumping the data
    *
    */
    virtual void dump(const char *fname)
    {
        if (entry != NULL)
        {
            char filename[256];
            if (strlen(fname) == 0)
            {
                sprintf(filename, "db_%03d.txt", entries - 1);
            }
            else
            {
                sprintf(filename, "%s_%03d.txt", fname, entries - 1);
            }

            FILE *f = fopen(filename, "w");

            for (int pos = 0; pos < entries - 1; pos++)
            {
                if (entry[pos].Parray != NULL)
                {
                    for (int i = 0; i < entry[pos].ndimParray; i++)
                    {
                        fprintf(f, "%20.16lf ", entry[pos].Parray[i]);
                    }
                }
                if (entry[pos].Garray != NULL)
                {
                    fprintf(f, "%20.16lf ", entry[pos].Fvalue);
                    int i;
                    for (i = 0; i < entry[pos].ndimGarray - 1; i++)
                    {
                        fprintf(f, "%20.16lf ", entry[pos].Garray[i]);
                    }
                    fprintf(f, "%20.16lf\n", entry[pos].Garray[i]);
                }
                else
                {
                    fprintf(f, "%20.16lf\n", entry[pos].Fvalue);
                }
            }

            fclose(f);
        }
    };

    virtual void dump()
    {
        dump("");
    };
};

template <class T>
void database<T>::init(int nsize1)
{
    if (entry == NULL)
    {
        try
        {
            entry = new T[nsize1];
        }
        catch (const std::system_error &e)
        {
            std::cout << " System error with code " << e.code() << " meaning " << e.what() << std::endl;
        }
        for (int i = 0; i < nsize1; i++)
        {
            entry[i] = (T)0;
        }
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
        if (entry[pos].Parray != NULL)
        {
            try
            {
                delete[] entry[pos].Parray;
            }
            catch (const std::system_error &e)
            {
                std::cout << " System error with code " << e.code() << " meaning " << e.what() << std::endl;
            }
        }

        try
        {
            entry[pos].Parray = new double[ndimParray];
        }
        catch (const std::system_error &e)
        {
            std::cout << " System error with code " << e.code() << " meaning " << e.what() << std::endl;
        }
    }
    entry[pos].ndimParray = ndimParray;

    for (int i = 0; i < ndimParray; i++)
    {
        entry[pos].Parray[i] = Parray[i];
    }

    entry[pos].Fvalue = Fvalue;

    if (ndimGarray > entry[pos].ndimGarray)
    {
        if (entry[pos].Garray != NULL)
        {
            try
            {
                delete[] entry[pos].Garray;
            }
            catch (const std::system_error &e)
            {
                std::cout << " System error with code " << e.code() << " meaning " << e.what() << std::endl;
            }
        }

        try
        {
            entry[pos].Garray = new double[ndimGarray];
        }
        catch (const std::system_error &e)
        {
            std::cout << " System error with code " << e.code() << " meaning " << e.what() << std::endl;
        }
    }
    entry[pos].ndimGarray = ndimGarray;

    for (int i = 0; i < ndimGarray; i++)
    {
        entry[pos].Garray[i] = Garray[i];
    }

    entry[pos].surrogate = surrogate;
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

struct cgdb_t : database<cgdbp_t>
{
};

struct db_t : database<dbp_t>
{
    virtual void print()
    {
        if (entry != NULL)
        {
            std::cout << "---- database priniting ----" << std::endl;

            for (int pos = 0; pos < entries; pos++)
            {
                if (entry[pos].Parray != NULL)
                {
                    int j;
                    std::cout << "ENTRY"
                              << std::setw(5) << pos << " : POINT(";
                    for (j = 0; j < entry[pos].ndimParray - 1; j++)
                    {
                        std::cout << std::setw(20) << entry[pos].Parray[j] << ", ";
                    }
                    std::cout << std::setw(20) << entry[pos].Parray[j] << ") Fvalue="
                              << std::setw(20) << entry[pos].Fvalue << " Surrogate="
                              << std::setw(20) << entry[pos].surrogate << std::endl;
                }
                if (entry[pos].Garray != NULL)
                {
                    int i;
                    std::cout << "Garray=[";
                    for (i = 0; i < entry[pos].ndimGarray - 1; i++)
                    {
                        std::cout << std::setw(20) << entry[pos].Garray[i] << ", ";
                    }
                    std::cout << std::setw(20) << entry[pos].Garray[i] << "]" << std::endl;
                }
            }

            std::cout << "----------------------------" << std::endl;
        }
    };
};

struct resdb_t : database<resdbp_t>
{
};

#endif