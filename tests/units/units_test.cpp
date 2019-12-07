#include "units/units.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *
 * \brief Tests units class
 *
 */
TEST(units_test, HandlesConstruction)
{
    // Create an instance of the units class
    umuq::units u("electron");

    EXPECT_EQ(u.getUnitStyle(), umuq::UnitStyle::ELECTRON);
    EXPECT_EQ(u.getUnitStyleName(), "ELECTRON");

    // Set the convert style
    EXPECT_TRUE(u.convertToStyle("metal"));

    double l = 1.0;
    u.convert<umuq::UnitType::Length>(l);

    // 1 Bohr = 0.529177208 Angstrom
    EXPECT_DOUBLE_EQ(l, 0.529177208);

    // 1 Hartree = 27.211396 eV
    std::vector<double> EnergyVector{1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};

    u.convert<umuq::UnitType::Energy>(EnergyVector);
    int n = 1;
    for (auto i : EnergyVector)
    {
        EXPECT_DOUBLE_EQ(i, n * 27.211396);
        n++;
    }

    EXPECT_EQ(umuq::getUnitStyleName(umuq::UnitStyle::METAL), "METAL");

    std::vector<double> LengthVector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    EXPECT_TRUE(umuq::convert<umuq::LengthUnit>(LengthVector, umuq::UnitStyle::ELECTRON, umuq::UnitStyle::METAL));

    EXPECT_FALSE(umuq::convert<umuq::UnitStyle>(LengthVector, umuq::UnitStyle::ELECTRON, umuq::UnitStyle::METAL));

    n = 1;
    for (auto i : LengthVector)
    {
        EXPECT_DOUBLE_EQ(i, n * 0.529177208);
        n++;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
