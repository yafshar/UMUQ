#ifndef UMHBM_RUNINFO_H
#define UMHBM_RUNINFO_H
#include "../io/io.hpp"
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
                  CoefVar(nullptr),
                  p(nullptr),
                  currentuniques(nullptr),
                  logselection(nullptr),
                  acceptance(nullptr),
                  SS(nullptr),
                  meantheta(nullptr){};

    bool save(const char *fileName = "runinfo.txt")
    {
        io f;
        if (f.openFile(fileName, f.out | f.app))
        {
            std::fstream &fs = f.getFstream();
            fs << "Gen=\n";
            fs << Gen << "\n";



            f.closeFile();
            return true;
        }
        return false;
    }
};

void save_runinfo()
{
    int i, j;

    /* allocate and initialize runinfo */
    FILE *f = fopen("runinfo.txt", "w");

    fprintf(f, "Gen=\n");
    fprintf(f, "%d\n", runinfo.Gen);

    fprintf(f, "CoefVar[%d]=\n", data.MaxStages);
    for (i = 0; i < data.MaxStages; i++)
        fprintf(f, "%.16lf\n", runinfo.CoefVar[i]);

    fprintf(f, "p[%d]=\n", data.MaxStages);
    for (i = 0; i < data.MaxStages; i++)
        fprintf(f, "%.16lf\n", runinfo.p[i]);

    fprintf(f, "currentuniques[%d]=\n", data.MaxStages);
    for (i = 0; i < data.MaxStages; i++)
        fprintf(f, "%d\n", runinfo.currentuniques[i]);

    fprintf(f, "logselection[%d]=\n", data.MaxStages);
    for (i = 0; i < data.MaxStages; i++)
        fprintf(f, "%.16lf\n", runinfo.logselection[i]);

    fprintf(f, "acceptance[%d]=\n", data.MaxStages);
    for (i = 0; i < data.MaxStages; i++)
        fprintf(f, "%.16lf\n", runinfo.acceptance[i]);

    fprintf(f, "SS[%d][%d]=\n", data.Nth, data.Nth);
    for (i = 0; i < data.Nth; i++)
        for (j = 0; j < data.Nth; j++)
            fprintf(f, "%.16lf\n", runinfo.SS[i][j]);

    fprintf(f, "meantheta[%d][%d]\n", data.MaxStages, data.Nth);
    for (i = 0; i < data.MaxStages; i++)
        for (j = 0; j < data.Nth; j++)
            fprintf(f, "%.16lf\n", runinfo.meantheta[i][j]);

    fclose(f);
}

int load_runinfo()
{
    int i, j;
    char header[256];

    /* allocate and initialize runinfo */
    FILE *f = fopen("runinfo.txt", "r");
    if (f == NULL)
        return 1;

    fscanf(f, "%s", header);
    fscanf(f, "%d", &runinfo.Gen);

    fscanf(f, "%s", header);
    for (i = 0; i < data.MaxStages; i++)
        fscanf(f, "%lf\n", &runinfo.CoefVar[i]);

    fscanf(f, "%s", header);
    for (i = 0; i < data.MaxStages; i++)
        fscanf(f, "%lf\n", &runinfo.p[i]);

    fscanf(f, "%s", header);
    for (i = 0; i < data.MaxStages; i++)
        fscanf(f, "%d\n", &runinfo.currentuniques[i]);

    fscanf(f, "%s", header);
    for (i = 0; i < data.MaxStages; i++)
        fscanf(f, "%lf\n", &runinfo.logselection[i]);

    fscanf(f, "%s", header);
    for (i = 0; i < data.MaxStages; i++)
        fscanf(f, "%lf\n", &runinfo.acceptance[i]);

    fscanf(f, "%s", header);
    for (i = 0; i < data.Nth; i++)
        for (j = 0; j < data.Nth; j++)
            fscanf(f, "%lf\n", &runinfo.SS[i][j]);

    fscanf(f, "%s", header);
    for (i = 0; i < data.MaxStages; i++)
        for (j = 0; j < data.Nth; j++)
            fscanf(f, "%lf\n", &runinfo.meantheta[i][j]);

    fclose(f);

    return 0;
}

#endif
