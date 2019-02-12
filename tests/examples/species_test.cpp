#include "core/core.hpp"
#include "../examples/speciesname.hpp"
#include "gtest/gtest.h"

/*!
 * \ingroup Test_Module
 *  
 * \brief Tests species class 
 *
 */
TEST(species_test, HandlesConstruction)
{
	umuq::species Species;

	EXPECT_EQ(Species.getNumberOfSpecies(), 139);

	EXPECT_EQ(Species.getSpeciesName(umuq::SpeciesID::Li), "Li");
	EXPECT_EQ(Species.getSpeciesID("Li"), static_cast<int>(umuq::SpeciesID::Li));

	EXPECT_EQ(Species.getSpeciesName(umuq::SpeciesID::U), "U");
	EXPECT_EQ(Species.getSpeciesID("U"), static_cast<int>(umuq::SpeciesID::U));

	EXPECT_EQ(Species.getSpeciesName(umuq::SpeciesID::user18), "unknown");
	EXPECT_EQ(Species.getSpeciesID("user18"), static_cast<int>(umuq::SpeciesID::user18));

	EXPECT_TRUE(Species.getSpeciesID("NaCl3") < static_cast<int>(umuq::SpeciesID::electron));

	EXPECT_EQ(Species.getSpeciesName(umuq::SpeciesID::Al), "Al");
	EXPECT_EQ(Species.getSpeciesID("Al"), static_cast<int>(umuq::SpeciesID::Al));

	auto Al = Species.getSpecies("Al");
	EXPECT_DOUBLE_EQ(Al.mass, 26.98154);

	auto Ba = Species.getSpecies("Ba");
	EXPECT_DOUBLE_EQ(Ba.mass, 137.327);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}