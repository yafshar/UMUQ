#ifndef UMHBM_DATATYPE_H
#define UMHBM_DATATYPE_H

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <system_error>

#include <cstdlib> //qsort
#include <cstdio>  //fopen, fgets, sscanf, sprintf
#include <cstring> //strlen, strstr, strtok

#include "../io/io.hpp"

#include "../misc/parser.hpp"
#include "../misc/array.hpp"

#include "../numerics/eigenmatrix.hpp"

/*! \file TMCMC_datatype.hpp
*   \brief Data types and helper structures & classes.
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
* \param lb                  generic lower bound
* \param ub                  generic upper bound
* \param TolCOV
* \param bbeta
* \param seed
* \param options
* \param sampling_type       sampling type which can be 0: uniform, 1: gaussian, 2: file 
* \param prior_type          prior type which can be  0: lognormal, 1: gaussian
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
class data_t
{
  public:
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
    * \brief constructor for the default variables
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
        try
        {
            lowerbound = new double[Nth];
            upperbound = new double[Nth];
            prior_mu = new double[Nth];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        }

        int e = Nth;
        while (e--)
        {
            lowerbound[e] = 0;
            upperbound[e] = 0;
            prior_mu[e] = 0;
        }

        try
        {
            prior_sigma = new double[Nth * Nth];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        }

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

        try
        {
            Num = new int[MaxStages];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        }

        e = MaxStages;
        while (e--)
            Num[e] = PopSize;

        try
        {
            local_cov = new double *[PopSize];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
        }

        for (int i = 0; i < PopSize; i++)
        {
            try
            {
                local_cov[i] = new double[Nth * Nth];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            }

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
    *  \brief read the input file fname
    *
    * \param fname              name of the input file
    * \return true on success
    */
    bool read(const char *fname);
    bool read()
    {
        // read the tmcmc.par file for setting the input variables
        return read("tmcmc.par");
    };

    //! destructor
    /*!
    *  \brief destructor 
    *    
    */
    ~data_t()
    {
        destroy();
    };

    /*!
    *  \brief destroy created memory 
    *
    */
    void destroy();
};

// read the input file fname for setting the input variables
bool data_t::read(const char *fname)
{
    // We use an IO object to open and read a file
    io u;

    if (u.openFile(fname))
    {
        // We need a parser object to parse
        parser p;

        int probdim = Nth;
        int maxgens = MaxStages;
        int datanum = PopSize;
        bool linit;

        while (u.readLine())
        {
            if (u.emptyLine())
            {
                continue;
            }

            char *line = u.getLine();
            char **lineArg = u.getLineArg();

            // Parse the line into line arguments
            p.parse(line, lineArg);

            std::string str(lineArg[0]);

            if (str == "Nth")
            {
                p.parse(lineArg[1], Nth);
            }
            else if (str == "MaxStages")
            {
                p.parse(lineArg[1], MaxStages);
            }
            else if (str == "PopSize")
            {
                p.parse(lineArg[1], PopSize);
            }
            else if (str == "TolCOV")
            {
                p.parse(lineArg[1], TolCOV);
            }
            else if (str == "bbeta")
            {
                p.parse(lineArg[1], bbeta);
            }
            else if (str == "seed")
            {
                p.parse(lineArg[1], seed);
            }
            else if (str == "opt.MaxIter")
            {
                p.parse(lineArg[1], options.MaxIter);
            }
            else if (str == "opt.Tol")
            {
                p.parse(lineArg[1], options.Tol);
            }
            else if (str == "opt.Display")
            {
                p.parse(lineArg[1], options.Display);
            }
            else if (str == "opt.Step")
            {
                p.parse(lineArg[1], options.Step);
            }
            else if (str == "prior_type")
            {
                p.parse(lineArg[1], prior_type);
            }
            else if (str == "prior_count")
            {
                p.parse(lineArg[1], prior_count);
            }
            else if (str == "iplot")
            {
                p.parse(lineArg[1], iplot);
            }
            else if (str == "icdump")
            {
                p.parse(lineArg[1], icdump);
            }
            else if (str == "ifdump")
            {
                p.parse(lineArg[1], ifdump);
            }
            else if (str == "Bdef")
            {
                p.parse(lineArg[1], lb);
                p.parse(lineArg[2], ub);
            }
            else if (str == "MinChainLength")
            {
                p.parse(lineArg[1], MinChainLength);
            }
            else if (str == "MaxChainLength")
            {
                p.parse(lineArg[1], MaxChainLength);
            }
            else if (str == "use_local_cov")
            {
                p.parse(lineArg[1], use_local_cov);
            }
        }

        linit = !(probdim == Nth && maxgens == MaxStages && datanum == PopSize && lowerbound != NULL);
        if (linit)
        {
            if (lowerbound != NULL)
            {
                delete[] lowerbound;
            }

            try
            {
                lowerbound = new double[Nth];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }

            if (upperbound != NULL)
            {
                delete[] upperbound;
            }

            try
            {
                upperbound = new double[Nth];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }
        }

        int n = Nth;
        int found;
        while (n--)
        {
            u.rewindFile();

            found = 0;
            std::string strt("B" + std::to_string(n));

            while (u.readLine())
            {
                if (u.emptyLine())
                {
                    continue;
                }

                char *line = u.getLine();
                char **lineArg = u.getLineArg();

                p.parse(line, lineArg);

                std::string str(lineArg[0]);

                if (str == strt)
                {
                    p.parse(lineArg[1], lowerbound[n]);
                    p.parse(lineArg[2], upperbound[n]);
                    found = 1;
                    break;
                }
            }

            if (!found)
            {
                lowerbound[n] = lb; /* Bdef value or Default LB */
                upperbound[n] = ub; /* Bdef value of Default UB */
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

                try
                {
                    prior_mu = new double[Nth];
                }
                catch (std::bad_alloc &e)
                {
                    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                    std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                    return false;
                }
            }

            u.rewindFile();
            found = 0;

            while (u.readLine())
            {
                if (u.emptyLine())
                {
                    continue;
                }

                char *line = u.getLine();
                char **lineArg = u.getLineArg();

                p.parse(line, lineArg);

                std::string str(lineArg[0]);

                if (str == "prior_mu")
                {
                    for (n = 0; n < Nth; n++)
                    {
                        p.parse(lineArg[n + 1], prior_mu[n]);
                    }
                    found = 1;
                    break;
                }
            }

            if (!found)
            {
                n = Nth;
                while (n--)
                {
                    prior_mu[n] = 0.0;
                }
            }

            if (linit)
            {
                /* new, parse prior_sigma */
                if (prior_sigma != NULL)
                {
                    delete[] prior_sigma;
                }

                try
                {
                    prior_sigma = new double[Nth * Nth];
                }
                catch (std::bad_alloc &e)
                {
                    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                    std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                    return false;
                }
            }

            u.rewindFile();
            found = 0;

            while (u.readLine())
            {
                if (u.emptyLine())
                {
                    continue;
                }

                char *line = u.getLine();
                char **lineArg = u.getLineArg();

                p.parse(line, lineArg);

                std::string str(lineArg[0]);

                if (str == "prior_sigma")
                {
                    for (n = 0; n < Nth * Nth; n++)
                    {
                        p.parse(lineArg[n + 1], prior_sigma[n]);
                    }
                    found = 1;
                    break;
                }
            }

            if (!found)
            {
                int i, j, k;
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
            try
            {
                compositeprior_distr = new double[Nth];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }

            if (linit)
            {
                if (prior_mu != NULL)
                {
                    delete[] prior_mu;
                }

                try
                {
                    prior_mu = new double[Nth];
                }
                catch (std::bad_alloc &e)
                {
                    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                    std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                    return false;
                }

                if (prior_sigma != NULL)
                {
                    delete[] prior_sigma;
                }

                try
                {
                    prior_sigma = new double[Nth * Nth];
                }
                catch (std::bad_alloc &e)
                {
                    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                    std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                    return false;
                }
            }

            n = Nth;
            while (n--)
            {
                u.rewindFile();

                found = 0;
                std::string strt("C" + std::to_string(n));

                while (u.readLine())
                {
                    if (u.emptyLine())
                    {
                        continue;
                    }

                    char *line = u.getLine();
                    char **lineArg = u.getLineArg();

                    p.parse(line, lineArg);

                    std::string str(lineArg[0]);

                    if (str == strt)
                    {
                        p.parse(lineArg[1], compositeprior_distr[n]);
                        p.parse(lineArg[2], lowerbound[n]);
                        p.parse(lineArg[3], upperbound[n]);
                        found = 1;
                        break;
                    }
                }

                if (!found)
                {
                    compositeprior_distr[n] = 0;
                    lowerbound[n] = lb; /* Bdef value or Default LB */
                    upperbound[n] = ub; /* Bdef value of Default UB */
                }
            }
        }

        /* new, parse auxil_size and auxil_data */
        u.rewindFile();
        found = 0;

        while (u.readLine())
        {
            if (u.emptyLine())
            {
                continue;
            }

            char *line = u.getLine();
            char **lineArg = u.getLineArg();

            p.parse(line, lineArg);

            std::string str(lineArg[0]);

            if (str == "auxil_size")
            {
                p.parse(lineArg[1], auxil_size);
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

            try
            {
                auxil_data = new double[auxil_size];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }

            u.rewindFile();

            found = 0;

            while (u.readLine())
            {
                if (u.emptyLine())
                {
                    continue;
                }

                char *line = u.getLine();
                char **lineArg = u.getLineArg();

                p.parse(line, lineArg);

                std::string str(lineArg[0]);

                if (str == "auxil_data")
                {
                    for (n = 0; n < auxil_size; n++)
                    {
                        p.parse(lineArg[n + 1], auxil_data[n]);
                    }
                    found = 1;
                    break;
                }
            }

            if (!found)
            {
                int i;
                for (i = 0; i < auxil_size; i++)
                {
                    auxil_data[i] = 0;
                }
            }
        }

        u.closeFile();

        if (linit)
        {
            if (Num != NULL)
            {
                delete[] Num;
            }

            try
            {
                Num = new int[MaxStages];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }

            n = MaxStages;
            while (n--)
            {
                Num[n] = PopSize;
            }
            LastNum = PopSize;

            if (local_cov != NULL)
            {
                n = PopSize;
                while (n--)
                {
                    if (local_cov[n] != NULL)
                    {
                        delete[] local_cov[n];
                    }
                }

                delete[] local_cov;
            }

            try
            {
                local_cov = new double *[PopSize];
            }
            catch (std::bad_alloc &e)
            {
                std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                return false;
            }
            for (n = 0; n < PopSize; n++)
            {
                try
                {
                    local_cov[n] = new double[Nth * Nth];
                }
                catch (std::bad_alloc &e)
                {
                    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
                    std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
                    return false;
                }
                for (int i = 0, l = 0; i < Nth; i++)
                {
                    for (int j = 0; j < Nth; j++, l++)
                    {
                        if (i == j)
                        {
                            local_cov[n][l] = 1;
                        }
                        else
                        {
                            local_cov[n][l] = 0;
                        }
                    }
                }
            }
        }
        return true;
    }
    return false;
};

/*!
*  \brief destroy the allocated memory 
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
    * constructor for the default variables it initializes to zero
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
* \tparam T type of database structure
* \param entry
* \param entries an integer argument shows the size of entry
* \param m A mutex object
*/
template <class T>
class database
{
  public:
    T *entry;
    int entries;

    std::unique_ptr<T[]> dataMemory;
    ArrayWrapper<T> data;

  private:
    pthread_mutex_t m;

  public:
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
    bool init(int nsize1);
    bool init(int nsize1, int nsize2)
    {
        return init(nsize1 * nsize2);
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
    bool update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray, int surrogate);
    bool update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray)
    {
        return update(*Parray, ndimParray, Fvalue, *Garray, ndimGarray, 0);
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
    inline bool dump(const char *fname)
    {
        if (entry != NULL)
        {
            char fileName[256];
            if (strlen(fname) == 0)
            {
                sprintf(fileName, "db_%03d.txt", entries - 1);
            }
            else
            {
                sprintf(fileName, "%s_%03d.txt", fname, entries - 1);
            }

            io f;
            if (f.openFile(fileName, f.out | f.trunc))
            {
                auto fs = f.getFstream();

                double **tmp = nullptr;
                int nRows = 2 + (int)(entry[0].Garray != NULL);
                tmp = new double *[3];

                for (int pos = 0; pos < entries - 1; pos++)
                {
                    tmp[0] = entry[pos].Parray;
                    tmp[1] = &entry[pos].Fvalue;
                    tmp[2] = entry[pos].Garray;

                    int nCols = entry[pos].ndimParray + 1 + entry[pos].ndimGarray;

                    if (!saveMatrix<double>(fs, tmp, nRows, nCols, 2))
                    {
                        return false;
                    }
                }
                f.closeFile();

                delete[] tmp;
                return true;
            }
            return false;
        }
        return false;
    };

    inline bool dump()
    {
        return dump("");
    };

    /*!
    * /brief function for loading the data
    *
    */
    bool load(const char *fname)
    {
        // if (entry != NULL)
        // {
        //     char fileName[256];
        //     if (strlen(fname) == 0)
        //     {
        //         sprintf(fileName, "db_%03d.txt", entries - 1);
        //     }
        //     else
        //     {
        //         sprintf(fileName, "%s_%03d.txt", fname, entries - 1);
        //     }

        //     io f;
        //     if (f.openFile(fileName, f.in))
        //     {
        //         auto fs = f.getFstream();

        //         double **tmp = nullptr;

        //         int nRows = 2 + (int)(entry[0].Garray != NULL);
        //         tmp = new double *[3];

        //         for (int pos = 0; pos < entries - 1; pos++)
        //         {
        //             tmp[0] = entry[pos].Parray;
        //             tmp[1] = &entry[pos].Fvalue;
        //             tmp[2] = entry[pos].Garray;

        //             int nCols = entry[pos].ndimParray + 1;

        //             loadMatrix<double>(fs, tmp, nRows, nCols, 1);
        //         }
        //         f.closeFile();

        //         delete[] tmp;
        //     }
        //     else
        //     {
        //         return false;
        //     }
        // }
        // return false;
    }
};

template <class T>
bool database<T>::init(int nsize1)
{
    if (entry == NULL)
    {
        try
        {
            entry = new T[nsize1];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        };

        for (int i = 0; i < nsize1; i++)
        {
            entry[i] = (T)0;
        }
        return true;
    }
    return false;
}

template <class T>
bool database<T>::update(double *Parray, int ndimParray, double Fvalue, double *Garray, int ndimGarray, int surrogate)
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
            delete[] entry[pos].Parray;
        }

        try
        {
            entry[pos].Parray = new double[ndimParray];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
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
            delete[] entry[pos].Garray;
        }

        try
        {
            entry[pos].Garray = new double[ndimGarray];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }
    }
    entry[pos].ndimGarray = ndimGarray;

    for (int i = 0; i < ndimGarray; i++)
    {
        entry[pos].Garray[i] = Garray[i];
    }

    entry[pos].surrogate = surrogate;
    return true;
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
inline void database<T>::sort(sort_t *list)
{
    qsort(list, entries, sizeof(sort_t), compar_desc);
}

struct cgdb_t : database<cgdbp_t>
{
};

struct db_t : database<dbp_t>
{
    // virtual void print()
    // {
    //     if (entry != NULL)
    //     {
    //         std::cout << "---- database priniting ----" << std::endl;

    //         for (int pos = 0; pos < entries; pos++)
    //         {
    //             if (entry[pos].Parray != NULL)
    //             {
    //                 int j;
    //                 std::cout << "ENTRY"
    //                           << std::setw(5) << pos << " : POINT(";
    //                 for (j = 0; j < entry[pos].ndimParray - 1; j++)
    //                 {
    //                     std::cout << std::setw(20) << entry[pos].Parray[j] << ", ";
    //                 }
    //                 std::cout << std::setw(20) << entry[pos].Parray[j] << ") Fvalue="
    //                           << std::setw(20) << entry[pos].Fvalue << " Surrogate="
    //                           << std::setw(20) << entry[pos].surrogate << std::endl;
    //             }
    //             if (entry[pos].Garray != NULL)
    //             {
    //                 int i;
    //                 std::cout << "Garray=[";
    //                 for (i = 0; i < entry[pos].ndimGarray - 1; i++)
    //                 {
    //                     std::cout << std::setw(20) << entry[pos].Garray[i] << ", ";
    //                 }
    //                 std::cout << std::setw(20) << entry[pos].Garray[i] << "]" << std::endl;
    //             }
    //         }

    //         std::cout << "----------------------------" << std::endl;
    //     }
    // };
};

struct resdb_t : database<resdbp_t>
{
};

#endif