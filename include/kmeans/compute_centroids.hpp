#ifndef KMEANS_COMPUTE_CENTROIDS_HPP
#define KMEANS_COMPUTE_CENTROIDS_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sanisizer/sanisizer.hpp"

#include "Matrix.hpp"
#include "utils.hpp"

namespace kmeans {

namespace internal {

template<class Matrix_, typename Float_>
void compute_centroid(const Matrix_& data, Float_* const center) {
    const auto ndim = data.num_dimensions();
    const auto nobs = data.num_observations();
    std::fill_n(center, ndim, 0);

    auto work = data.new_known_extractor(static_cast<I<decltype(nobs)> >(0), nobs);
    for (I<decltype(nobs)> i = 0; i < nobs; ++i) {
        const auto dptr = work->get_observation();
        for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
            center[d] += static_cast<Float_>(dptr[d]); // cast for consistent precision regardless of the matrix datatype.
        }
    }

    for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
        center[d] /= nobs;
    }
}

template<class Matrix_, typename Cluster_, typename Float_>
void compute_centroids(const Matrix_& data, const Cluster_ ncenters, Float_* const centers, const Cluster_* clusters, const std::vector<Index<Matrix_> >& sizes) {
    const auto nobs = data.num_observations();
    const auto ndim = data.num_dimensions();
    std::fill_n(centers, sanisizer::product_unsafe<std::size_t>(ndim, ncenters), 0);

    auto work = data.new_known_extractor(static_cast<I<decltype(nobs)> >(0), nobs);
    for (I<decltype(nobs)> obs = 0; obs < nobs; ++obs) {
        const auto curclust = clusters[obs];
        const auto mine = work->get_observation();
        for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
            centers[sanisizer::nd_offset<std::size_t>(d, ndim, curclust)] += static_cast<Float_>(mine[d]); // cast for consistent precision regardless of the matrix datatype.
        }
    }

    for (I<decltype(ncenters)> cen = 0; cen < ncenters; ++cen) {
        const auto s = sizes[cen];
        if (s) {
            for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
                centers[sanisizer::nd_offset<std::size_t>(d, ndim, cen)] /= s;
            }
        }
    }
}

}

}

#endif
