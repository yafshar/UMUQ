#include "numerics/vnearestneighbors.hpp"

#include "datatype/eigendatatype.hpp"
#include "gtest/gtest.h"
#include "io/io.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"

TEST(vnearestneighbors_test, HandlesConstruct) {
  int const ndataPoints = 1000;
  int const nDimension = 2;
  int const accuracyOrder = 2;

  umuq::vNearestNeighbor<double> VNN(ndataPoints, nDimension, accuracyOrder);

  EXPECT_EQ(VNN.numNearestNeighbors(), 3);

  auto v = std::move(VNN);

  EXPECT_EQ(v.numNearestNeighbors(), 3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
