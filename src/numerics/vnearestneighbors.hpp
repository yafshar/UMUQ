#ifndef UMUQ_VNEARESTNEIGHBORS_H
#define UMUQ_VNEARESTNEIGHBORS_H

#include <cstddef>
#include <utility>
#include <vector>

#include "core/core.hpp"
#include "datatype/distancetype.hpp"
#include "datatype/eigendatatype.hpp"
#include "eigenlib.hpp"
#include "knearestneighborbase.hpp"
#include "polynomials.hpp"

namespace {
template <typename RealType>
static inline int monomialsize(int const dim, int const PolynomialOrder) {
  umuq::polynomials::polynomialBase<RealType> p(dim, PolynomialOrder);
  return p.monomialsize();
}
}  // namespace

namespace umuq {

template <typename RealType, class PolynomialType = polynomial<RealType>>
class vNearestNeighbor
    : public kNearestNeighborBase<RealType, flann::L2<RealType>> {
 public:
  vNearestNeighbor(int const ndataPoints, int const nDimension,
                   int const accuracyOrder);
  vNearestNeighbor(int const ndataPoints, int const nqueryPoints,
                   int const nDimension, int const accuracyOrder);
  vNearestNeighbor(vNearestNeighbor<RealType, PolynomialType> &&other) = default;
  vNearestNeighbor(vNearestNeighbor<RealType, PolynomialType> const &other) = default;
  vNearestNeighbor<RealType, PolynomialType> &operator=(vNearestNeighbor<RealType, PolynomialType> &&other) = default;
  ~vNearestNeighbor() = default;

 private:
  /*! Order of accuracy */
  int order;

  /*! Polynomial object with polynomial degree of \f$ r - 1 \f$ */
  PolynomialType poly;
};

template <typename RealType, class PolynomialType>
vNearestNeighbor<RealType, PolynomialType>::vNearestNeighbor(
    int const ndataPoints, int const nDimension, int const accuracyOrder)
    : kNearestNeighborBase<RealType, flann::L2<RealType>>(
          ndataPoints, nDimension,
          monomialsize<RealType>(nDimension, accuracyOrder - 1)),
      order(accuracyOrder),
      poly(nDimension, accuracyOrder - 1) {
  if (!(unrolledIncrement == 0 || unrolledIncrement == 4 ||
        unrolledIncrement == 6 || unrolledIncrement == 8 ||
        unrolledIncrement == 10 || unrolledIncrement == 12)) {
    std::string msg = "The unrolled increment value of ";
    msg += "'unrolledIncrement' is not correctly set!";
    UMUQFAIL(msg);
  }
  if (this->nDataPoints <
      static_cast<std::size_t>(this->nNearestNeighborsToFind)) {
    std::string msg = "Not enough points to create K ";
    msg += "nearest neighbors for each point !";
    UMUQFAIL(msg);
  }
}

template <typename RealType, class PolynomialType>
vNearestNeighbor<RealType, PolynomialType>::vNearestNeighbor(
    int const ndataPoints, int const nqueryPoints, int const nDimension,
    int const accuracyOrder)
    : kNearestNeighborBase<RealType, flann::L2<RealType>>(
          ndataPoints, nqueryPoints, nDimension,
          monomialsize<RealType>(nDimension, accuracyOrder - 1)),
      order(accuracyOrder),
      poly(nDimension, accuracyOrder - 1) {
  if (!(unrolledIncrement == 0 || unrolledIncrement == 4 ||
        unrolledIncrement == 6 || unrolledIncrement == 8 ||
        unrolledIncrement == 10 || unrolledIncrement == 12)) {
    std::string msg = "The unrolled increment value of ";
    msg += "'unrolledIncrement' is not correctly set!";
    UMUQFAIL(msg);
  }
  if (this->nDataPoints <
      static_cast<std::size_t>(this->nNearestNeighborsToFind)) {
    std::string msg = "Not enough points to create K ";
    msg += "nearest neighbors for each point !";
    UMUQFAIL(msg);
  }
}

// template <typename RealType, class PolynomialType>
// vNearestNeighbor<RealType, PolynomialType>::vNearestNeighbor(
//     vNearestNeighbor<RealType, PolynomialType> &&other)
//     : nDataPoints(other.nDataPoints),
//       nQueryDataPoints(other.nQueryDataPoints),
//       dataDimension(other.dataDimension),
//       nNearestNeighborsToFind(other.nNearestNeighborsToFind),
//       indices_ptr(std::move(other.indices_ptr)),
//       dists_ptr(std::move(other.dists_ptr)),
//       indices(std::move(other.indices)),
//       dists(std::move(other.dists)),
//       the_same(other.the_same),
//       withCovariance(other.withCovariance),
//       order(other.order),
//       poly(std::move(other.poly)) {}

// template <typename RealType, class PolynomialType>
// vNearestNeighbor<RealType, PolynomialType>::vNearestNeighbor(
//     vNearestNeighbor<RealType, PolynomialType> const &other)
//     : nDataPoints(other.nDataPoints),
//       nQueryDataPoints(other.nQueryDataPoints),
//       dataDimension(other.dataDimension),
//       nNearestNeighborsToFind(other.nNearestNeighborsToFind),
//       indices_ptr(new int[nQueryDataPoints * nNearestNeighborsToFind]),
//       dists_ptr(new DataType[nQueryDataPoints * nNearestNeighborsToFind]),
//       indices(indices_ptr.get(), nQueryDataPoints, nNearestNeighborsToFind),
//       dists(dists_ptr.get(), nQueryDataPoints, nNearestNeighborsToFind),
//       the_same(other.the_same),
//       withCovariance(other.withCovariance),
//       order(other.order),
//       poly(dataDimension, order - 1) {
//   {
//     int *From = other.indices_ptr.get();
//     int *To = indices_ptr.get();
//     std::copy(From, From + nQueryDataPoints * nNearestNeighborsToFind, To);
//   }
//   {
//     DataType *From = other.dists_ptr.get();
//     DataType *To = dists_ptr.get();
//     std::copy(From, From + nQueryDataPoints * nNearestNeighborsToFind, To);
//   }
// }

// template <typename RealType, class PolynomialType>
// vNearestNeighbor<RealType, PolynomialType>
//     &vNearestNeighbor<RealType, PolynomialType>::operator=(
//         vNearestNeighbor<RealType, PolynomialType> &&other) {
//   nDataPoints = std::move(other.nDataPoints);
//   nQueryDataPoints = std::move(other.nQueryDataPoints);
//   dataDimension = std::move(other.dataDimension);
//   nNearestNeighborsToFind = std::move(other.nNearestNeighborsToFind);
//   indices_ptr = std::move(other.indices_ptr);
//   dists_ptr = std::move(other.dists_ptr);
//   indices = std::move(other.indices);
//   dists = std::move(other.dists);
//   the_same = std::move(other.the_same);
//   withCovariance = std::move(other.withCovariance);
//   order = std::move(other.order);
//   poly = std::move(other.poly);

//   return static_cast<vNearestNeighbor<RealType, PolynomialType> &>(
//       kNearestNeighborBase<RealType, flann::L2<RealType>>::operator=(
//           std::move(other)));
// }

}  // namespace umuq

#endif  // UMUQ_VNEARESTNEIGHBORS