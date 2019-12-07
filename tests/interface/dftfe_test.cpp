#include "interface/dftfe.hpp"
#include "environment.hpp"
#include "gtest/gtest.h"

// Tests dftfe
TEST(dftfe_test, HandlesConstruction)
{
    // Create an instance of the dftfe object
    umuq::dftfe dft;

    // Set the input file
    dft.setFullFileName("./", "AlDimer_bondlength4d");

    EXPECT_TRUE(dft.getSpeciesInformation());

    EXPECT_TRUE(dft.getNumberSpecies() == 2);
    EXPECT_TRUE(dft.getNumberSpeciesTypes() == 1);
    EXPECT_TRUE(dft.getTotalNumberRunFiles() == 3);
    EXPECT_TRUE(dft.getTotalNumberSteps() == 3);

    auto s = dft.getSpecies();
    EXPECT_TRUE(s.size() == dft.getNumberSpeciesTypes());
    EXPECT_TRUE(s[0].name == "Al");
    EXPECT_DOUBLE_EQ(s[0].mass, 26.98154);

    std::string FileName;
    for (int i = 0; i < dft.getTotalNumberSteps(); i++)
    {
        FileName = "COORDS" + std::to_string(i);
        std::remove(FileName.c_str());
        FileName = "XCOORDS" + std::to_string(i);
        std::remove(FileName.c_str());
        FileName = "FORCE" + std::to_string(i);
        std::remove(FileName.c_str());
    }

    EXPECT_TRUE(dft.dump());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new umuq::torcEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    // Adds UMUQ listener; Google Test owns this pointer
    listeners.Append(new umuq::UMUQEventListener);

    return RUN_ALL_TESTS();
}
