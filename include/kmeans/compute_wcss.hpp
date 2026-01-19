#ifndef KMEANS_COMPUTE_WCSS_HPP
#define KMEANS_COMPUTE_WCSS_HPP

#include <algorithm>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file compute_wcss.hpp
 * @brief Compute within-cluster sum of squares.
 */

namespace kmeans {

/**
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centers and output.
 *
 * @param data A matrix containing data for each observation. 
 * @param num_centers Number of cluster centers.
 * @param[in] centers Pointer to an array of length equal to the product of `num_centers` and `data.num_dimensions()`.
 * This contains a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers.
 * The `j`-th column should contain the initial centroid location for cluster `j`.
 * @param[in] clusters Pointer to an array of length equal to the number of observations (from `data.num_observations()`).
 * This should contain the 0-based cluster assignment for each observation.
 * Specifically, each entry is an index that refers to a column of `centers` and is no greater than `num_centers`.
 * @param[out] wcss Pointer to an array of length equal to `num_centers`.
 * On output, this will contain the within-cluster sum of squares for each cluster.
 */
template<class Matrix_, typename Cluster_, typename Float_>
void compute_wcss(const Matrix_& data, const Cluster_ num_centers, const Float_* const centers, const Cluster_* const clusters, Float_* const wcss) {
    const auto nobs = data.num_observations();
    const auto ndim = data.num_dimensions();
    std::fill_n(wcss, num_centers, 0);

    auto work = data.new_known_extractor(static_cast<I<decltype(nobs)> >(0), nobs);
    for (I<decltype(nobs)> obs = 0; obs < nobs; ++obs) {
        const auto curdata = work->get_observation();
        const auto cen = clusters[obs];

        Float_& curwcss = wcss[cen];
        for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
            const auto curcenter = centers[sanisizer::nd_offset<std::size_t>(d, ndim, cen)];
            const Float_ delta = static_cast<Float_>(curdata[d]) - curcenter; // cast for consistent precision regardless of the input data type.
            curwcss += delta * delta;
        }
    }
}

}

#endif
