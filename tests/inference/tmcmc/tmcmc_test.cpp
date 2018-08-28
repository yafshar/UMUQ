#include "core/core.hpp"
#include "core/environment.hpp"
#include "inference/tmcmc/tmcmc.hpp"
#include "gtest/gtest.h"

//! Tests tmcmc
TEST(tmcmc_test, HandlesConstruction)
{
	tmcmc<double> t;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new torcEnvironment);

    return RUN_ALL_TESTS();
}