#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <iomanip>

#include "polynomial.hpp"
#include "eigen.hpp"
#include "flann.hpp"

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

    double x[d];
    for (i = 0; i < d; i++)
    {
        x[i] = 2.0;
    }

    double *value;
    value = p.monomial_value(d, r, alpha, x);

    for (i = 0; i < p.binomial_coefficient(d + r, r); i++)
    {
        std::cout << i << " " << value[i] << std::endl;
    }

    delete[] alpha;
    delete[] value;

    EMatrixXd A;
    A.resize(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
    std::cout << "Here is the matrix A:" << std::endl;
    std::cout << A << std::endl;

    EVectorXd B;
    B.resize(3);
    B << 3, 3, 4;
    std::cout << "Here is the vector B:" << std::endl;
    std::cout << B << std::endl;

    EVectorXd X = A.fullPivLu().solve(B);
    std::cout << "The solution is: " << std::endl;
    std::cout << X << std::endl;

    auto relative_error = (A * X - B).norm() / B.norm(); // norm() is L2 norm
    std::cout << "The relative error is:" << std::endl;
    std::cout << relative_error << std::endl;

    EVectorXd Y = A.partialPivLu().solve(B);
    std::cout << "The solution is: " << std::endl;
    std::cout << Y << std::endl;
    
    relative_error = (A * Y - B).norm() / B.norm(); // norm() is L2 norm
    std::cout << "The relative error is:" << std::endl;
    std::cout << relative_error << std::endl;
    return 0;
}
