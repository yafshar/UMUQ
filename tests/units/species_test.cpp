#include "core/core.hpp"
#include "units/speciesname.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *
 * \brief Tests species class
 *
 */
TEST(species_test, HandlesConstruction)
{
    // Create an instance of the species object
    umuq::species Species;

    EXPECT_EQ(Species.getNumberOfSpecies(), 140);

    EXPECT_EQ(Species.getSpeciesName(umuq::SpeciesID::Li), "Li");
    EXPECT_EQ(Species.getSpeciesID("Li"), umuq::SpeciesID::Li);
    EXPECT_EQ(Species.getSpeciesIndex("Li"), static_cast<int>(umuq::SpeciesID::Li));

    EXPECT_EQ(Species.getSpeciesName(umuq::SpeciesID::U), "U");
    EXPECT_EQ(Species.getSpeciesID("U"), umuq::SpeciesID::U);
    EXPECT_EQ(Species.getSpeciesIndex("U"), static_cast<int>(umuq::SpeciesID::U));

    EXPECT_EQ(Species.getSpeciesName(umuq::SpeciesID::user18), "unknown");
    EXPECT_EQ(Species.getSpeciesID("user18"), umuq::SpeciesID::user18);
    EXPECT_EQ(Species.getSpeciesIndex("user18"), static_cast<int>(umuq::SpeciesID::user18));

    // If element or species does not exist it's ID is unknown
    EXPECT_EQ(Species.getSpeciesID("NaCl3") , umuq::SpeciesID::unknown);
    EXPECT_EQ(Species.getSpeciesIndex("NaCl3") , static_cast<int>(umuq::SpeciesID::unknown));

    EXPECT_EQ(Species.getSpeciesName(umuq::SpeciesID::Al), "Al");
    EXPECT_EQ(Species.getSpeciesID("Al"), umuq::SpeciesID::Al);
    EXPECT_EQ(Species.getSpeciesIndex("Al"), static_cast<int>(umuq::SpeciesID::Al));

    auto Al = Species.getSpecies("Al");
    EXPECT_DOUBLE_EQ(Al.mass, 26.98154);

    auto Ba = Species.getSpecies("Ba");
    EXPECT_DOUBLE_EQ(Ba.mass, 137.327);

    auto u = Species.getSpecies("NewElement");
    EXPECT_EQ(u.name, "unknown");
    EXPECT_TRUE(u.mass > 300.);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}