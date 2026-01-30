#ifndef KMEANS_REMOVE_UNUSED_CENTERS_HPP
#define KMEANS_REMOVE_UNUSED_CENTERS_HPP

#include <vector>
#include <cstddef>
#include <algorithm>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file remove_unused_centers.hpp
 * @brief Remove unused k-means centroids.
 */

namespace kmeans {

/**
 * Remove unused k-means centroids from the `centers` array filled by `Refine::run()`.
 * Specifically, clusters are relabelled so that all empty clusters have higher cluster indices in `clusters` than the non-empty clusters.
 * On output, `clusters` will contain all and only integers in `[0, N)` where `N` is the number of non-empty clusters.
 * This ensures that downstream applications do not waste time processing `num_centers` clusters when some of them are empty.
 *
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 *
 * @param num_dimensions Number of dimensions.
 * @param num_observations Number of observations.
 * @param[in,out] clusters Pointer to an array of length `num_observations`.
 * On input, `clusters[i]` should contain the index of the cluster assigned to observation `i`, typically from `Refine::run()`.
 * On output, `clusters[i]` contains a possibly-remapped index that will be less than the number of non-empty clusters (returned by this function).
 * @param num_centers Number of cluster centers.
 * @param[in,out] centers Pointer to an array of length equal to the product of `num_dimensions` and `num_centers`.
 * On input, this should contain a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers,
 * i.e., the `j`-th column defines the centroid for cluster `j`, which may be present in `clusters`.
 * On output, the columns are rearranged to match the remapped indices in `clusters`.
 * Coordinates for empty clusters are left in an arbitrary state.
 * @param sizes Vector of length equal to `num_centers`, containing the number of observations in each cluster.
 * On input, the `j`-th value should be equal to the frequency of cluster `j` in `clusters`, typically from `Details::sizes`.
 * On output, the values are rearranged to match the remapped indices in `clusters`.
 * Sizes for empty clusters are left in an arbitrary state.
 *
 * @return Number of non-empty clusters.
 * If this is equal to `num_centers`, this function is a no-op.
 */
template<typename Index_, typename Cluster_, typename Float_>
Cluster_ remove_unused_centers(
    const std::size_t num_dimensions,
    const Index_ num_observations,
    Cluster_* const clusters,
    const Cluster_ num_centers,
    Float_* const centers,
    std::vector<Index_>& sizes
) {
    bool has_zero = false;
    for (Cluster_ c = 0; c < num_centers; ++c) {
        if (sizes[c] == 0) {
            has_zero = true;
            break;
        }
    }
    if (!has_zero) {
        return num_centers;
    }

    auto remapping = sanisizer::create<std::vector<Index_> >(num_centers);
    Cluster_ remaining = 0;
    for (Cluster_ c = 0; c < num_centers; ++c) {
        if (sizes[c]) {
            remapping[c] = remaining;
            if (remaining != c) {
                std::copy_n(
                    centers + sanisizer::product_unsafe<std::size_t>(c, num_dimensions),
                    num_dimensions,
                    centers + sanisizer::product_unsafe<std::size_t>(remaining, num_dimensions)
                );
                sizes[remaining] = sizes[c];
            }
            ++remaining;
        }
    }

    for (Index_ o = 0; o < num_observations; ++o) {
        clusters[o] = remapping[clusters[o]];
    }

    return remaining;
}

/**
 * Overload of `remove_unused_clusters()` where, for all empty clusters, centroids are set to `fill` and sizes are set to zero.
 *
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 *
 * @param num_dimensions Number of dimensions.
 * @param num_observations Number of observations.
 * @param[in,out] clusters Pointer to an array of length `num_observations`.
 * On input, `clusters[i]` should contain the index of the cluster assigned to observation `i`, typically from `Refine::run()`.
 * On output, `clusters[i]` contains a possibly-remapped index that will be less than the number of non-empty clusters (returned by this function).
 * @param num_centers Number of cluster centers.
 * @param[in,out] centers Pointer to an array of length equal to the product of `num_dimensions` and `num_centers`.
 * On input, this should contain a column-major matrix where rows correspond to dimensions and columns correspond to cluster centers,
 * i.e., the `j`-th column defines the centroid for cluster `j`, which may be present in `clusters`.
 * On output, the columns are rearranged to match the remapped indices in `clusters`.
 * Coordinates for empty clusters are set to `fill`.
 * @param sizes Vector of length equal to `num_centers`, containing the number of observations in each cluster.
 * On input, the `j`-th value should be equal to the frequency of cluster `j` in `clusters`, typically from `Details::sizes`.
 * On output, the values are rearranged to match the remapped indices in `clusters`.
 * Sizes for empty clusters are set to zero.
 * @param fill Fill value for the centers of unused clusters.
 *
 * @return Number of non-empty clusters, see `remove
 * If this is equal to `num_centers`, this function is a no-op.
 */
template<typename Index_, typename Cluster_, typename Float_>
Cluster_ remove_unused_centers(
    const std::size_t num_dimensions,
    const Index_ num_observations,
    Cluster_* const clusters,
    const Cluster_ num_centers,
    Float_* const centers,
    std::vector<Index_>& sizes,
    Float_ fill
) {
    const auto remaining = remove_unused_centers(num_dimensions, num_observations, clusters, num_centers, centers, sizes);

    std::fill(sizes.begin() + remaining, sizes.end(), 0);

    std::fill(
        centers + sanisizer::product_unsafe<std::size_t>(remaining, num_dimensions),
        centers + sanisizer::product_unsafe<std::size_t>(num_centers, num_dimensions),
        fill
    );

    return remaining;
}

}

#endif
