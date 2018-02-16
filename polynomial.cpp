#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <iomanip>

#include "polynomial.h"

int main()
{

    int d = 3;
    int r = 2;
    int *alpha = NULL;

    polynomial p;

    p.monomial_basis(d, r, alpha);

    std::cout << " d =  " << d << std::endl;
    std::cout << " r =  " << r << std::endl;
    
    int i, n;
    n = 0;
    for (i = 0; i < p.binomial_coefficient(d + r, r); i++)
    {
        std::cout << "  " << std::setw(2) << i << "    ";
        for (int j = 0; j < d; j++)
        {
            std::cout << std::setw(2) << alpha[n];
            n++;
        }
        std::cout << std::endl;
    }

    double x[d]={1.0,2.0,3.0};
    //for (i = 0; i < d; i++)
    //{
    //    x[i] = 2.0;
    //}

    double *value;
    value = p.monomial_value(d, r, alpha, x);

    for (i = 0; i < p.binomial_coefficient(d + r, r); i++)
    {
        std::cout << i << " " << value[i] << std::endl;
    }

    delete[] alpha;
    delete[] value;
    return 0;
}
