#include <iostream>
#include <stdio.h>

#include "../src/data_type/TMCMC_datatype.hpp"
#include "gtest/gtest.h"

#define BUFLEN 1024

// Tests parse
TEST(datatype, HandlesConstruction)
{
    data_t d1;

    EXPECT_EQ(0, d1.Nth);
    EXPECT_EQ(0, d1.MaxStages);
    EXPECT_EQ(0, d1.PopSize);
    EXPECT_EQ(0, d1.auxil_size);
    EXPECT_EQ(0, d1.MinChainLength);
    EXPECT_EQ(1e6, d1.MaxChainLength);
    EXPECT_EQ(0, d1.lb);
    EXPECT_EQ(0, d1.ub);
    EXPECT_EQ(1.0, d1.TolCOV);
    EXPECT_EQ(0.2, d1.bbeta);
    EXPECT_EQ(280675, d1.seed);
    EXPECT_EQ(0, d1.prior_type);
    EXPECT_EQ(0, d1.prior_count);
    EXPECT_EQ(0, d1.iplot);
    EXPECT_EQ(1, d1.icdump);
    EXPECT_EQ(0, d1.ifdump);
    EXPECT_EQ(0, d1.LastNum);
    EXPECT_EQ(0, d1.use_proposal_cma);
    EXPECT_EQ(0, d1.use_local_cov);
    EXPECT_EQ(0, d1.local_scale);
    
    EXPECT_EQ(100, d1.options.MaxIter);
    EXPECT_EQ(1e-6, d1.options.Tol);
    EXPECT_EQ(0, d1.options.Display);
    EXPECT_EQ(1e-5, d1.options.Step);

    data_t d2(4,20,1024);

    EXPECT_EQ(4, d2.Nth);
    EXPECT_EQ(20, d2.MaxStages);
    EXPECT_EQ(1024, d2.PopSize);
    EXPECT_EQ(0, d2.auxil_size);
    EXPECT_EQ(0, d2.MinChainLength);
    EXPECT_EQ(1e6, d2.MaxChainLength);
    EXPECT_EQ(-6, d2.lb);
    EXPECT_EQ(+6, d2.ub);
    EXPECT_EQ(1.0, d2.TolCOV);
    EXPECT_EQ(0.2, d2.bbeta);
    EXPECT_EQ(280675, d2.seed);
    EXPECT_EQ(0, d2.prior_type);
    EXPECT_EQ(0, d2.prior_count);
    EXPECT_EQ(0, d2.iplot);
    EXPECT_EQ(1, d2.icdump);
    EXPECT_EQ(0, d2.ifdump);
    EXPECT_EQ(1024, d2.LastNum);
    EXPECT_EQ(0, d2.use_proposal_cma);
    EXPECT_EQ(0, d2.use_local_cov);
    EXPECT_EQ(0, d2.local_scale);

    EXPECT_EQ(100, d2.options.MaxIter);
    EXPECT_EQ(1e-6, d2.options.Tol);
    EXPECT_EQ(0, d2.options.Display);
    EXPECT_EQ(1e-5, d2.options.Step);
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}