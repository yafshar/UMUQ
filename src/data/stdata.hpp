#ifndef UMUQ_STDATA_H
#define UMUQ_STDATA_H

#include "../core/core.hpp"
#include "../io/io.hpp"
#include "../misc/parser.hpp"

/*! \file stdata.hpp
*  \brief stream Data type
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
class stdata
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
    stdata() : Nth(0),
               MaxStages(0),
               PopSize(0),
               lowerbound(nullptr),
               upperbound(nullptr),
               compositeprior_distr(nullptr),
               prior_mu(nullptr),
               prior_sigma(nullptr),
               auxil_size(0),
               auxil_data(nullptr),
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
               Num(nullptr),
               LastNum(0),
               use_proposal_cma(0),
               init_mean(nullptr),
               local_cov(nullptr),
               use_local_cov(0),
               local_scale(0){};

    //! constructor
    /*!
    *  \brief constructor for the default input variables
    *    
    */
    stdata(int probdim, int maxgens, int datanum) : Nth(probdim),
                                                    MaxStages(maxgens),
                                                    PopSize(datanum),
                                                    lowerbound(nullptr),
                                                    upperbound(nullptr),
                                                    compositeprior_distr(nullptr),
                                                    prior_mu(nullptr),
                                                    prior_sigma(nullptr),
                                                    auxil_size(0),
                                                    auxil_data(nullptr),
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
                                                    Num(nullptr),
                                                    LastNum(datanum),
                                                    use_proposal_cma(0),
                                                    init_mean(nullptr),
                                                    local_cov(nullptr),
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

        {
            int e = Nth;
            while (e--)
            {
                lowerbound[e] = 0;
                upperbound[e] = 0;
                prior_mu[e] = 0;
            }
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

        {
            int e = MaxStages;
            while (e--)
            {
                Num[e] = PopSize;
            }
        }
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
    bool read(const char *fname = "tmcmc.par");

    //! destructor
    /*!
    *  \brief destructor 
    *    
    */
    ~stdata()
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
bool stdata::read(const char *fname)
{
    // We use an IO object to open and read a file
    io f;

    if (f.openFile(fname, f.in))
    {
        // We need a parser object to parse
        parser p;

        int probdim = Nth;
        int maxgens = MaxStages;
        int datanum = PopSize;
        bool linit;

        //read each line in the file and skip all the commented and empty line with the defaukt comment "#"
        while (f.readLine())
        {
            // Parse the line into line arguments
            p.parse(f.getLine());

            if (p.at<std::string>(0) == "Nth")
            {
                Nth = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "MaxStages")
            {
                MaxStages = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "PopSize")
            {
                PopSize = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "TolCOV")
            {
                TolCOV = p.at<double>(1);
            }
            else if (p.at<std::string>(0) == "bbeta")
            {
                bbeta = p.at<double>(1);
            }
            else if (p.at<std::string>(0) == "seed")
            {
                seed = p.at<long>(1);
            }
            else if (p.at<std::string>(0) == "opt.MaxIter")
            {
                options.MaxIter = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "opt.Tol")
            {
                options.Tol = p.at<double>(1);
            }
            else if (p.at<std::string>(0) == "opt.Display")
            {
                options.Display = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "opt.Step")
            {
                options.Step = p.at<double>(1);
            }
            else if (p.at<std::string>(0) == "prior_type")
            {
                prior_type = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "prior_count")
            {
                prior_count = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "iplot")
            {
                iplot = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "icdump")
            {
                icdump = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "ifdump")
            {
                ifdump = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "Bdef")
            {
                lb = p.at<double>(1);
                ub = p.at<double>(2);
            }
            else if (p.at<std::string>(0) == "MinChainLength")
            {
                MinChainLength = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "MaxChainLength")
            {
                MaxChainLength = p.at<int>(1);
            }
            else if (p.at<std::string>(0) == "use_local_cov")
            {
                use_local_cov = p.at<int>(1);
            }
        }

        linit = !(probdim == Nth && maxgens == MaxStages && datanum == PopSize && lowerbound != nullptr);
        if (linit)
        {
            if (lowerbound != nullptr)
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

            if (upperbound != nullptr)
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
            f.rewindFile();

            found = 0;
            std::string strt("B" + std::to_string(n));

            while (f.readLine())
            {
                p.parse(f.getLine());

                if (p.at<std::string>(0) == strt)
                {
                    lowerbound[n] = p.at<double>(1);
                    upperbound[n] = p.at<double>(2);
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
                if (prior_mu != nullptr)
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

            f.rewindFile();
            found = 0;

            while (f.readLine())
            {
                p.parse(f.getLine());

                if (p.at<std::string>(0) == "prior_mu")
                {
                    for (n = 0; n < Nth; n++)
                    {
                        prior_mu[n] = p.at<double>(n + 1);
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
                if (prior_sigma != nullptr)
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

            f.rewindFile();
            found = 0;

            while (f.readLine())
            {
                p.parse(f.getLine());

                if (p.at<std::string>(0) == "prior_sigma")
                {
                    for (n = 0; n < Nth * Nth; n++)
                    {
                        prior_sigma[n] = p.at<double>(n + 1);
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
            if (compositeprior_distr != nullptr)
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
                if (prior_mu != nullptr)
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

                if (prior_sigma != nullptr)
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
                f.rewindFile();

                found = 0;
                std::string strt("C" + std::to_string(n));

                while (f.readLine())
                {
                    p.parse(f.getLine());

                    if (p.at<std::string>(0) == strt)
                    {
                        compositeprior_distr[n] = p.at<double>(1);
                        lowerbound[n] = p.at<double>(2);
                        upperbound[n] = p.at<double>(3);
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
        f.rewindFile();
        found = 0;

        while (f.readLine())
        {
            p.parse(f.getLine());

            if (p.at<std::string>(0) == "auxil_size")
            {
                auxil_size = p.at<int>(1);
                found = 1;
                break;
            }
        }

        if (auxil_size > 0)
        {
            if (auxil_data != nullptr)
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

            f.rewindFile();

            found = 0;

            while (f.readLine())
            {
                p.parse(f.getLine());

                if (p.at<std::string>(0) == "auxil_data")
                {
                    for (n = 0; n < auxil_size; n++)
                    {
                        auxil_data[n] = p.at<double>(n + 1);
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

        f.closeFile();

        if (linit)
        {
            if (Num != nullptr)
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

            if (local_cov != nullptr)
            {
                n = PopSize;
                while (n--)
                {
                    if (local_cov[n] != nullptr)
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
void stdata::destroy()
{
    if (lowerbound != nullptr)
    {
        delete[] lowerbound;
        lowerbound = nullptr;
    }
    if (upperbound != nullptr)
    {
        delete[] upperbound;
        upperbound = nullptr;
    }
    if (compositeprior_distr != nullptr)
    {
        delete[] compositeprior_distr;
        compositeprior_distr = nullptr;
    }
    if (prior_mu != nullptr)
    {
        delete[] prior_mu;
        prior_mu = nullptr;
    }
    if (prior_sigma != nullptr)
    {
        delete[] prior_sigma;
        prior_sigma = nullptr;
    }
    if (auxil_data != nullptr)
    {
        delete[] auxil_data;
        auxil_data = nullptr;
    }
    if (Num != nullptr)
    {
        delete[] Num;
        Num = nullptr;
    }
    if (init_mean != nullptr)
    {
        delete[] * init_mean;
        delete[] init_mean;
        init_mean = nullptr;
    }
    if (local_cov != nullptr)
    {
        delete[] * local_cov;
        delete[] local_cov;
        local_cov = nullptr;
    }
}

#endif