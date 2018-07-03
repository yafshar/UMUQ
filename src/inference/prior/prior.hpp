#ifndef UMUQ_PRIOR_H
#define UMUQ_PRIOR_H

#include "../../core/core.hpp"


class priorDistribution
{

};



//TODO: define the struct Prior:
//typedef struct{
//	Density *d;
//	int Nd;
//} Prior;

void delete_density(Density *d);
void delete_prior(Density *d, int N);

double eval_density(Density d, double x);
double eval_log_density(Density d, double x);
double eval_random(Density d);

void print_density(Density d);
void print_priors(Density *d, int N);

double prior_pdf(Density *d, int N, double *x);
double prior_log_pdf(Density *d, int N, double *x);

void read_priors(const char *file, Density **p_priors, int *p_N);

void reassign_prior(Density *p, int Np, double *psi);
void new_prior_from_prior(Density **new_prior, Density *from_prior, int Npr);

void check_n(int N);

#endif
