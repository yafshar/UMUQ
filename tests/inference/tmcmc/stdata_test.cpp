#include "inference/tmcmc/stdata.hpp"
#include "gtest/gtest.h"

// Tests parse
TEST(streamdatatype, HandlesConstruction)
{
    umuq::tmcmc::stdata *d1 = new umuq::tmcmc::stdata;

    EXPECT_EQ(0, d1->nDim);
    EXPECT_EQ(0, d1->maxGenerations);
    EXPECT_EQ(0, d1->populationSize);
    EXPECT_EQ(0, d1->lastPopulationSize);
    EXPECT_EQ(0, d1->auxilSize);
    EXPECT_EQ(1, d1->minChainLength);
    EXPECT_EQ(1, d1->maxChainLength);
    EXPECT_EQ(280675, d1->seed);
    EXPECT_EQ(0, d1->samplingType);
    EXPECT_EQ(umuq::priorTypes::UNIFORM, d1->priorType);
    EXPECT_EQ(0, d1->iPlot);
    EXPECT_EQ(1, d1->saveData);
    EXPECT_EQ(0, d1->useCmaProposal);
    EXPECT_EQ(0, d1->useLocalCovariance);
    EXPECT_DOUBLE_EQ(static_cast<double>(1), d1->coefVarPresetThreshold);
    EXPECT_DOUBLE_EQ(static_cast<double>(0.2), d1->bbeta);
    EXPECT_EQ(100, d1->options.MaxIter);
    EXPECT_DOUBLE_EQ(static_cast<double>(1e-6), d1->options.Tolerance);
    EXPECT_EQ(0, d1->options.Display);
    EXPECT_DOUBLE_EQ(static_cast<double>(1e-5), d1->options.Step);

    delete d1;

    umuq::tmcmc::stdata d2(4, 20, 1024);

    EXPECT_EQ(4, d2.nDim);
    EXPECT_EQ(20, d2.maxGenerations);
    EXPECT_EQ(1024, d2.populationSize);
    EXPECT_EQ(1024, d2.lastPopulationSize);
    EXPECT_EQ(0, d2.auxilSize);
    EXPECT_EQ(1, d2.minChainLength);
    EXPECT_EQ(1, d2.maxChainLength);
    EXPECT_EQ(280675, d2.seed);
    EXPECT_EQ(0, d2.samplingType);
    EXPECT_EQ(umuq::priorTypes::UNIFORM, d2.priorType);
    EXPECT_EQ(0, d2.iPlot);
    EXPECT_EQ(1, d2.saveData);
    EXPECT_EQ(0, d2.useCmaProposal);
    EXPECT_EQ(0, d2.useLocalCovariance);
    EXPECT_DOUBLE_EQ(1.0, d2.coefVarPresetThreshold);
    EXPECT_DOUBLE_EQ(0.2, d2.bbeta);
    EXPECT_DOUBLE_EQ(static_cast<double>(1), d2.coefVarPresetThreshold);
    EXPECT_DOUBLE_EQ(static_cast<double>(0.2), d2.bbeta);
    EXPECT_EQ(100, d2.options.MaxIter);
    EXPECT_DOUBLE_EQ(static_cast<double>(1e-6), d2.options.Tolerance);
    EXPECT_EQ(0, d2.options.Display);
    EXPECT_DOUBLE_EQ(static_cast<double>(1e-5), d2.options.Step);
}

// Tests load input file
TEST(streamdatatype, HandlesIO)
{
    umuq::tmcmc::stdata d1;

    std::string fileName = "./tmcmc/test.txt";

    EXPECT_TRUE(d1.load(fileName));

    EXPECT_EQ(4, d1.nDim);
    EXPECT_EQ(20, d1.maxGenerations);
    EXPECT_EQ(5000, d1.populationSize);
    EXPECT_EQ(5000, d1.lastPopulationSize);
    EXPECT_EQ(0, d1.auxilSize);
    EXPECT_EQ(1, d1.minChainLength);
    EXPECT_EQ(1, d1.maxChainLength);
    EXPECT_DOUBLE_EQ(static_cast<double>(1), d1.coefVarPresetThreshold);
    EXPECT_DOUBLE_EQ(static_cast<double>(0.04), d1.bbeta);
    EXPECT_EQ(280675, d1.seed);
    EXPECT_EQ(0, d1.samplingType);
    EXPECT_EQ(umuq::priorTypes::UNIFORM, d1.priorType);
    EXPECT_EQ(0, d1.iPlot);
    EXPECT_EQ(1, d1.saveData);
    EXPECT_EQ(0, d1.useCmaProposal);
    EXPECT_EQ(0, d1.useLocalCovariance);
    EXPECT_EQ(1000, d1.options.MaxIter);
    EXPECT_DOUBLE_EQ(static_cast<double>(1e-12), d1.options.Tolerance);
    EXPECT_EQ(1, d1.options.Display);
    EXPECT_DOUBLE_EQ(static_cast<double>(1e-4), d1.options.Step);
    EXPECT_DOUBLE_EQ(0.05, d1.lowerBound[0]);
    EXPECT_DOUBLE_EQ(3.0, d1.lowerBound[1]);
    EXPECT_DOUBLE_EQ(6.01, d1.lowerBound[2]);
    EXPECT_DOUBLE_EQ(0.0001, d1.lowerBound[3]);
    EXPECT_DOUBLE_EQ(10.0, d1.upperBound[0]);
    EXPECT_DOUBLE_EQ(4.0, d1.upperBound[1]);
    EXPECT_DOUBLE_EQ(15.0, d1.upperBound[2]);
    EXPECT_DOUBLE_EQ(1.0, d1.upperBound[3]);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
