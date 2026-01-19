#ifndef KMEANS_IS_EDGE_CASE_HPP
#define KMEANS_IS_EDGE_CASE_HPP

#include <numeric>
#include <algorithm>
#include <vector>

#include "sanisizer/sanisizer.hpp"

#include "Details.hpp"
#include "Matrix.hpp"
#include "compute_centroids.hpp"
#include "utils.hpp"

namespace kmeans {

namespace internal {

template<typename Index_, typename Cluster_>
bool is_edge_case(const Index_ nobs, const Cluster_ ncenters) {
    return ncenters <= 1 || sanisizer::is_greater_than_or_equal(ncenters, nobs);
}

template<class Matrix_, typename Cluster_, typename Float_>
Details<Index<Matrix_> > process_edge_case(const Matrix_& data, Cluster_ ncenters, Float_* centers, Cluster_* clusters) {
    const auto nobs = data.num_observations();

    if (ncenters == 1) {
        // All points in cluster 0.
        std::fill_n(clusters, nobs, 0);
        auto sizes = sanisizer::create<std::vector<I<decltype(nobs)> > >(1, nobs);
        compute_centroid(data, centers);
        return Details(std::move(sizes), 0, 0);

    } else if (sanisizer::is_greater_than_or_equal(ncenters, nobs)) {
        // Special case, each observation is a center.
        std::iota(clusters, clusters + nobs, static_cast<Cluster_>(0));
        auto sizes = sanisizer::create<std::vector<I<decltype(nobs)> > >(ncenters);
        std::fill_n(sizes.begin(), nobs, 1);

        const auto ndim = data.num_dimensions();
        auto work = data.new_known_extractor(static_cast<I<decltype(nobs)> >(0), nobs);
        for (I<decltype(nobs)> o = 0; o < nobs; ++o) {
            const auto ptr = work->get_observation();
            std::copy_n(ptr, ndim, centers + sanisizer::product_unsafe<std::size_t>(o, ndim));
        }

        return Details(std::move(sizes), 0, 0);

    } else { //i.e., ncenters == 0, provided is_edge_case is true.
        return Details<Index<Matrix_> >(0, 0);
    }
}

}

}

#endif
