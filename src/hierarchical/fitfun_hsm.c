#include "fitfun.h"

#include "gsl_headers.h"

#define FILE1 "../psi/final.txt"
#define FILE2 "data.txt"

#define BUFLEN 1024
#define NREC   10000
#define D_IND     1
#define N_IND     5
#define FAIL   -1e12


typedef struct ffdata_s {
    double *x;
    double logprior;
} ffdata_t;

static ffdata_t ffdata[NREC];

// number of data in data file : FILE2
static int     Nd=0;
static double *datax;
static double *datay;

#include "engine_tmcmc.h"
extern data_t data;

void fitfun_initialize() {

    printf("\nReading data from psi database \n");

    int PROBDIM = data.Nth;
    // psi has twice of theta dimension
    int PPROBDIM = PROBDIM*2;

    char filename[BUFLEN];
    snprintf(filename, BUFLEN, "%s",FILE1);

    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("\n %s  does not exist. Exiting...\n", filename);
        exit(1);
    }

    // count the number of lines in file and check if less than NREC
    char ch;
    int lines = 0;
    while (!feof(fp)) {
        ch = fgetc(fp);
        if (ch == '\n') lines++;
    }
    rewind(fp);

    if (lines < NREC) {
        printf("\n\n Error: Number of samples less than %d in file %s. Exit... \n\n", NREC, filename);
        exit(1);
    }

    for (int i = 0; i < NREC; i++)
        ffdata[i].x = (double*)malloc(PPROBDIM*sizeof(double));

    double dummy;

    for (int i = 0; i < NREC; i++) {
        for (int j = 0; j < PPROBDIM; j++)
            fscanf(fp, "%lf", &(ffdata[i].x[j]));

        fscanf(fp, "%lf", &dummy); //loglike
        fscanf(fp, "%lf", &dummy); //logprior
        for (int j=0; j< D_IND-1; j++)
            fscanf(fp, "%lf", &dummy);
        fscanf(fp, "%lf", &ffdata[i].logprior);
        for (int j= D_IND ; j< N_IND; j++)
            fscanf(fp, "%lf", &dummy);
    }
    fclose(fp);

    printf("\nSuccesfull reading data from psi database.\n\n");

    // open the data file
    snprintf(filename, BUFLEN, "%s", FILE2);

    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("\n %s  does not exist. Exiting...\n", filename);
        exit(1);
    }
    printf("\nData file %s is succesfully opened \n",filename);

    // read the number of lines of data file
    while (!feof(fp)) {
        ch = fgetc(fp);
        if (ch == '\n') Nd++;
    }
    rewind(fp);

    datax = (double*)malloc(Nd*sizeof(double));
    datay = (double*)malloc(Nd*sizeof(double));
    for(int i=0; i<Nd; i++){
        fscanf( fp, "%lf", &datax[i]);
        fscanf( fp, "%lf", &datay[i]);
    }

    printf("\n%d data succesfully read from %s \n",Nd,filename);

    fclose(fp);
}

void fitfun_finalize() {
    for (int i = 0; i < NREC; i++)
        free(ffdata[i].x);

    free(datax);
    free(datay);
}

// uniform distribution
inline double log_unifpdf(double x, double l, double u) {
    if ((l <= x) && (x <= u)) return -log(u-l);
    return NAN;
}

// uniform distribution
inline double unifpdf2(double x, double l, double len) {
    if ((l <= x) && (x <= l+len)) return 1.0/len;
    return 0;
}

// truncated normal distribution
inline double trnpdf(double x, double m, double s, double l, double u) {
    if (s > 0)
        return gsl_ran_gaussian_pdf(fabs(x-m), s) / (1.0 - gsl_cdf_gaussian_P(fabs(m-l), s) - gsl_cdf_gaussian_P(fabs(m-u), s));
    else
        return NAN;
}

// log-normal distribution = 1/x . 1/(s(2pi)^0.5) . exp(-(lnx-m)^2/(2s^2))
inline double log_lognpdf(double x, double m, double s) {
    if (s > 0)
        return -0.5*log(2*M_PI) - log(x) - log(s) -0.5*pow((log(x)-m)/s,2);
    else
        return NAN;
}

double log_nominator(double *theta, double *psi, int n) {
    double res = 0.0;
    for (int i = 0; i < n; i++) {
        res += log_lognpdf(theta[i], psi[2*i], psi[2*i+1]);
    }
    return res;
}

void my_model(double *x, double *y, double *c, double n){
    for(int i=0; i<n; i++)
        y[i] = c[0]*sin(c[1]*x[i]+c[2]);
}

// here x is theta and n number of those
double fitfun(double *x, int n, void *output, int *info) {

    double out;

    double sum = 0;
    for (int i = 0; i < NREC; i++) {
        double lognominator   = log_nominator(x, ffdata[i].x, n);
        double logdenominator = ffdata[i].logprior;
        sum += exp(lognominator - logdenominator) ;
    }

    if (sum == 0 || isnan(sum) || isinf(sum)) {
        out = FAIL;
        return out;
    }

    double res;
    // The experimental data are linked with the computational model through the
    // likelihood function, model assumption for the likelihood function involves
    // a Gaussian and covariance matrix of that may be a function of parameters
    double sigma2 = pow(x[n-1],2);

    double *y;
    y = (double*)malloc(Nd*sizeof(double));
    my_model(datax,y,x,Nd);

    double ssn=0.;
    for(int i=0; i<Nd; i++)
        ssn += pow( datay[i] - y[i], 2);

    free(y);

    res = Nd*log(2*M_PI) + Nd*log(sigma2) + ssn/sigma2;
    res *= -0.5;

    out = res - log(NREC) + log(sum) - (-52.294728);

    if (isinf(out) || isnan(out)) out=FAIL;

    return out;
}
