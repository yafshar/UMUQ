#include <iostream>
#include <cmath>

#include "../src/numerics/polynomial.hpp"
#include "gtest/gtest.h"

// Tests binomial coefficient c(n, k) of 0.
TEST(binomial_coefficient_test, HandlesZeroInput)
{
  polynomial p;

  EXPECT_EQ(1, p.binomial_coefficient(1, 0));
  EXPECT_EQ(0, p.binomial_coefficient(0, 1));
  EXPECT_EQ(1, p.binomial_coefficient(0, 0));
  EXPECT_EQ(1, p.binomial_coefficient(10, 0));
}

// Tests binomial coefficient c(n, k)
TEST(binomial_coefficient_test, HandlesOtherInput)
{
  polynomial p;

  EXPECT_EQ(10, p.binomial_coefficient(10, 1));
  EXPECT_EQ(1, p.binomial_coefficient(10, 10));
  EXPECT_EQ(6, p.binomial_coefficient(4, 2));
  EXPECT_EQ(10, p.binomial_coefficient(5, 2));
}

// Tests binomial coefficient c(n, k) of 0.
//
//   For example:
//       d = 2
//       r = 2
//
//       alpha[ 0],[ 1] = 0, 0 = x^0 y^0
//       alpha[ 2],[ 3] = 1, 0 = x^1 y^0
//       alpha[ 4],[ 5] = 0, 1 = x^0 y^1
//       alpha[ 6],[ 7] = 2, 0 = x^2 y^0
//       alpha[ 8],[ 9] = 1, 1 = x^1 y^1
//       alpha[10],[11] = 0, 2 = x^0 y^2
//
//       monomial_basis(2,2) = {1,    x,   y,  x^2, xy,  y^2}
//                     alpha = {0,0, 1,0, 0,1, 2,0, 1,1, 0,2}
//
//
//       monomial_basis(3,2) = {1,       x,     y,     z,    x^2,  xy,    xz,   y^2,    yz,    z^2  }
//                     alpha = {0,0,0, 1,0,0, 0,1,0, 0,0,1, 2,0,0 1,1,0, 1,0,1, 0,2,0, 0,1,1, 0,0,2 }
TEST(monomial_basis_test, HandlesInput)
{
  polynomial p;

  int dim;
  int degree;
  int *coeff;
  int num;

  dim = 2;
  degree = 2;

  p.monomial_basis(dim, degree, coeff);
  num = dim * p.binomial_coefficient(dim + degree, degree);
  EXPECT_EQ(12, num);

  int alpha[12] = {0, 0,
                   1, 0,
                   0, 1,
                   2, 0,
                   1, 1,
                   0, 2};

  int i;
  for (i = 0; i < num; i++)
  {
    EXPECT_EQ(alpha[i], coeff[i]);
  };

  delete[] coeff;
  coeff = NULL;

  dim = 3;

  p.monomial_basis(dim, degree, coeff);

  num = dim * p.binomial_coefficient(dim + degree, degree);
  EXPECT_EQ(30, num);

  int beta[30] = {0, 0, 0,
                  1, 0, 0,
                  0, 1, 0,
                  0, 0, 1,
                  2, 0, 0,
                  1, 1, 0,
                  1, 0, 1,
                  0, 2, 0,
                  0, 1, 1,
                  0, 0, 2};

  for (i = 0; i < num; i++)
  {
    EXPECT_EQ(beta[i], coeff[i]);
  };
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}