#ifndef UMHBM_RUNINFO_H
#define UMHBM_RUNINFO_H
#include "../io/io.hpp"
#include "../misc/parser.hpp"
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
    double *CoefVar;      /*[MAXGENS];*/
    double *p;            /*[MAXGENS];*/
    int *currentuniques;  /*[MAXGENS];*/
    double *logselection; /*[MAXGENS];*/
    double *acceptance;   /*[MAXGENS];*/
    double **SS;          /*[PROBDIM][PROBDIM];*/
    double **meantheta;   /*[MAXGENS][PROBDIM];*/

    int probdim;
    int maxgens;

    /*!
     * \brief constructor for the default variables
     *    
     */
    runinfo_t() { init(); }
    runinfo_t(int probdim_, int maxgens_) { init(probdim_, maxgens_); }

    /*!
     * \brief destructor
     *    
     */
    ~runinfo_t() { destroy(); }

    /*!
     * \brief  
     *
     */
    bool init(int probdim_ = 0, int maxgens_ = 0)
    {
        if (probdim_ == 0 || maxgens_ == 0)
        {
            probdim = 0;
            maxgens = 0;
            Gen = 0;
            CoefVar = nullptr;
            p = nullptr;
            currentuniques = nullptr;
            logselection = nullptr;
            acceptance = nullptr;
            SS = nullptr;
            meantheta = nullptr;

            return true;
        }

        probdim = probdim_;
        maxgens = maxgens_;
        Gen = 0;

        try
        {
            CoefVar = new double[maxgens]();
            p = new double[maxgens]();
            currentuniques = new int[maxgens]();
            logselection = new double[maxgens]();
            acceptance = new double[maxgens]();
            SS = new double *[probdim];
            meantheta = new double *[maxgens];
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }

        try
        {
            for (int i = 0; i < probdim; i++)
            {
                SS[i] = new double[probdim]();
            }
            for (int i = 0; i < maxgens; i++)
            {
                meantheta[i] = new double[probdim]();
            }
        }
        catch (std::bad_alloc &e)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Failed to allocate memory : " << e.what() << std::endl;
            return false;
        }

        //set the first value to a high number
        CoefVar[0] = 10.;

        return true;
    }

    /*!
     * \brief save the inofmration in a file @fileName
     * 
     * Write the runinfo data information to a file @fileName
     * 
     * \param fileName Name of the file (default name is runinfo.txt) for writing information 
     */
    bool save(const char *fileName = "runinfo.txt")
    {
        io f;
        if (f.openFile(fileName, f.out))
        {
            std::fstream &fs = f.getFstream();
            bool tmp = true;

            fs << std::fixed;
            fs << "Gen= " << Gen << "\n";

            fs << "CoefVar[" << maxgens << "]=\n";
            tmp |= f.saveMatrix<double>(CoefVar, maxgens);

            fs << "p[" << maxgens << "]=\n";
            tmp |= f.saveMatrix<double>(p, maxgens);

            fs << "currentuniques[" << maxgens << "]=\n";
            tmp |= f.saveMatrix<int>(currentuniques, maxgens);

            fs << "logselection[" << maxgens << "]=\n";
            tmp |= f.saveMatrix<double>(logselection, maxgens);

            fs << "acceptance[" << maxgens << "]=\n";
            tmp |= f.saveMatrix<double>(acceptance, maxgens);

            fs << "SS[" << probdim << "][" << probdim << "]=\n";
            tmp |= f.saveMatrix<double>(SS, probdim, probdim, 1);

            fs << "meantheta[" << maxgens << "][" << probdim << "]=\n";
            tmp |= f.saveMatrix<double>(meantheta, maxgens, probdim, 1);

            f.closeFile();
            return tmp;
        }
        return false;
    }

    /*!
     * \brief load inofmration from a file @fileName
     * 
     * Load the runinfo data information from a file @fileName
     * 
     * \param fileName Name of the file (default name is runinfo.txt) for reading information 
     */
    bool load(const char *fileName = "runinfo.txt")
    {
        io f;
        if (f.openFile(fileName, f.in))
        {
            parser prs;
            bool tmp = true;

            tmp |= f.readLine();
            tmp |= f.readLine();
            Gen = prs.at<int>(0);

            tmp |= f.readLine();
            tmp |= f.loadMatrix<double>(CoefVar, maxgens);

            tmp |= f.readLine();
            tmp |= f.loadMatrix<double>(p, maxgens);

            tmp |= f.readLine();
            tmp |= f.loadMatrix<int>(currentuniques, maxgens);

            tmp |= f.readLine();
            tmp |= f.loadMatrix<double>(logselection, maxgens);

            tmp |= f.readLine();
            tmp |= f.loadMatrix<double>(acceptance, maxgens);

            tmp |= f.readLine();
            tmp |= f.loadMatrix<double>(SS, probdim, probdim, 1);

            tmp |= f.readLine();
            tmp |= f.loadMatrix<double>(meantheta, maxgens, probdim, 1);

            f.closeFile();
            return tmp;
        }
        return false;
    }

    /*!
     * \brief destroy created memory 
     *
     */
    void destroy()
    {
        if (CoefVar != nullptr)
        {
            delete[] CoefVar;
            CoefVar = nullptr;
        }
        if (p != nullptr)
        {
            delete[] p;
            p = nullptr;
        }
        if (currentuniques != nullptr)
        {
            delete[] currentuniques;
            currentuniques = nullptr;
        }
        if (logselection != nullptr)
        {
            delete[] logselection;
            logselection = nullptr;
        }
        if (acceptance != nullptr)
        {
            delete[] acceptance;
            acceptance = nullptr;
        }
        if (SS != nullptr)
        {
            delete[] * SS;
            delete[] SS;
            SS = nullptr;
        }
        if (meantheta != nullptr)
        {
            delete[] * meantheta;
            delete[] meantheta;
            meantheta = nullptr;
        }
    }
};

#endif
