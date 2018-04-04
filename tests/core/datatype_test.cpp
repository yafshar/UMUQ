#include "core/datatype.hpp"
#include "gtest/gtest.h"

//! Tests parse
TEST(datatype, HandlesConstruction)
{
    data_t *d1 = new data_t;

    EXPECT_EQ(0, d1->Nth);
    EXPECT_EQ(0, d1->MaxStages);
    EXPECT_EQ(0, d1->PopSize);
    EXPECT_EQ(0, d1->auxil_size);
    EXPECT_EQ(0, d1->MinChainLength);
    EXPECT_EQ(1e6, d1->MaxChainLength);
    EXPECT_DOUBLE_EQ(0, d1->lb);
    EXPECT_DOUBLE_EQ(0, d1->ub);
    EXPECT_DOUBLE_EQ(1.0, d1->TolCOV);
    EXPECT_DOUBLE_EQ(0.2, d1->bbeta);
    EXPECT_EQ(280675, d1->seed);
    EXPECT_EQ(0, d1->prior_type);
    EXPECT_EQ(0, d1->prior_count);
    EXPECT_EQ(0, d1->iplot);
    EXPECT_EQ(1, d1->icdump);
    EXPECT_EQ(0, d1->ifdump);
    EXPECT_EQ(0, d1->LastNum);
    EXPECT_EQ(0, d1->use_proposal_cma);
    EXPECT_EQ(0, d1->use_local_cov);
    EXPECT_EQ(0, d1->local_scale);

    EXPECT_EQ(100, d1->options.MaxIter);
    EXPECT_DOUBLE_EQ(1e-6, d1->options.Tol);
    EXPECT_EQ(0, d1->options.Display);
    EXPECT_DOUBLE_EQ(1e-5, d1->options.Step);

    delete d1;

    data_t d2(4, 20, 1024);

    EXPECT_EQ(4, d2.Nth);
    EXPECT_EQ(20, d2.MaxStages);
    EXPECT_EQ(1024, d2.PopSize);
    EXPECT_EQ(0, d2.auxil_size);
    EXPECT_EQ(0, d2.MinChainLength);
    EXPECT_EQ(1e6, d2.MaxChainLength);
    EXPECT_DOUBLE_EQ(-6, d2.lb);
    EXPECT_DOUBLE_EQ(+6, d2.ub);
    EXPECT_DOUBLE_EQ(1.0, d2.TolCOV);
    EXPECT_DOUBLE_EQ(0.2, d2.bbeta);
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
    EXPECT_DOUBLE_EQ(1e-6, d2.options.Tol);
    EXPECT_EQ(0, d2.options.Display);
    EXPECT_DOUBLE_EQ(1e-5, d2.options.Step);

    d2.destroy();

    EXPECT_EQ(NULL, d2.lowerbound);
};

//! Tests read input file
TEST(datatype, HandlesIO)
{
    data_t d1;

    EXPECT_TRUE(d1.read("test.txt"));
    EXPECT_EQ(4, d1.Nth);
    EXPECT_EQ(20, d1.MaxStages);
    EXPECT_EQ(5000, d1.PopSize);
    EXPECT_EQ(0, d1.auxil_size);
    EXPECT_EQ(1, d1.MinChainLength);
    EXPECT_EQ(1, d1.MaxChainLength);
    EXPECT_DOUBLE_EQ(0, d1.lb);
    EXPECT_DOUBLE_EQ(0, d1.ub);
    EXPECT_DOUBLE_EQ(1.0, d1.TolCOV);
    EXPECT_DOUBLE_EQ(0.04, d1.bbeta);
    EXPECT_EQ(280675, d1.seed);
    EXPECT_EQ(0, d1.prior_type);
    EXPECT_EQ(1, d1.prior_count);
    EXPECT_EQ(0, d1.iplot);
    EXPECT_EQ(1, d1.icdump);
    EXPECT_EQ(1, d1.ifdump);
    EXPECT_EQ(5000, d1.LastNum);
    EXPECT_EQ(0, d1.use_proposal_cma);
    EXPECT_EQ(0, d1.use_local_cov);
    EXPECT_DOUBLE_EQ(0, d1.local_scale);

    EXPECT_EQ(1000, d1.options.MaxIter);
    EXPECT_DOUBLE_EQ(1e-12, d1.options.Tol);
    EXPECT_EQ(1, d1.options.Display);
    EXPECT_DOUBLE_EQ(1e-4, d1.options.Step);

    EXPECT_DOUBLE_EQ(0.05, d1.lowerbound[0]);
    EXPECT_DOUBLE_EQ(3.0, d1.lowerbound[1]);
    EXPECT_DOUBLE_EQ(6.01, d1.lowerbound[2]);
    EXPECT_DOUBLE_EQ(0.0001, d1.lowerbound[3]);
    EXPECT_DOUBLE_EQ(10.0, d1.upperbound[0]);
    EXPECT_DOUBLE_EQ(4.0, d1.upperbound[1]);
    EXPECT_DOUBLE_EQ(15.0, d1.upperbound[2]);
    EXPECT_DOUBLE_EQ(1.0, d1.upperbound[3]);  
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
