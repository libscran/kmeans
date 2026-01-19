#ifndef KMEANS_INITIALIZE_KMEANSPP_HPP
#define KMEANS_INITIALIZE_KMEANSPP_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>

#include "sanisizer/sanisizer.hpp"
#include "aarand/aarand.hpp"

#include "Initialize.hpp"
#include "Matrix.hpp"
#include "copy_into_array.hpp"
#include "parallelize.hpp"
#include "utils.hpp"

/**
 * @file InitializeKmeanspp.hpp
 *
 * @brief Class for kmeans++ initialization.
 */

namespace kmeans {

/**
 * Type of the pseudo-random number generator for `InitializeKmeanspp`.
 */
typedef std::mt19937_64 InitializeKmeansppRng;

/**
 * @brief Options for `InitializeKmeanspp`.
 */
struct InitializeKmeansppOptions {
    /**
     * Random seed to use to construct the PRNG prior to sampling.
     */
    typename InitializeKmeansppRng::result_type seed = sanisizer::cap<typename InitializeKmeansppRng::result_type>(6523);

    /** 
     * Number of threads to use.
     * The parallelization scheme is defined by `parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace InitializeKmeanspp_internal {

template<typename Float_, typename Index_, class Engine_>
Index_ weighted_sample(const std::vector<Float_>& cumulative, const std::vector<Float_>& mindist, Index_ nobs, Engine_& eng) {
    const auto total = cumulative.back();
    Index_ chosen_id = 0;

    do {
        const Float_ sampled_weight = total * aarand::standard_uniform<Float_>(eng);

        // Subtraction is safe as we already checked for valid ptrdiff in run_kmeanspp().
        chosen_id = std::lower_bound(cumulative.begin(), cumulative.end(), sampled_weight) - cumulative.begin();

        // We wrap this in a do/while to defend against edge cases where
        // ties are chosen. The most obvious of these is when you get a
        // `sampled_weight` of zero _and_ there exists a bunch of zeros at
        // the start of `cumulative`. One could also get unexpected ties
        // from limited precision in floating point comparisons, so we'll
        // just be safe and implement a loop here, in the same vein as
        // uniform01.
    } while (chosen_id == nobs || mindist[chosen_id] == 0);

    return chosen_id;
}

template<typename Index_, typename Float_, class Matrix_, typename Cluster_>
std::vector<Index_> run_kmeanspp(const Matrix_& data, Cluster_ ncenters, const typename InitializeKmeansppRng::result_type seed, const int nthreads) {
    const auto nobs = data.num_observations();
    const auto ndim = data.num_dimensions();
    auto mindist = sanisizer::create<std::vector<Float_> >(nobs, 1);

    auto cumulative = sanisizer::create<std::vector<Float_> >(nobs);
    sanisizer::can_ptrdiff<I<decltype(cumulative.begin())> >(nobs); // check that we can compute a ptrdiff for weighted_sample().

    std::vector<Index_> sofar;
    sofar.reserve(ncenters);
    InitializeKmeansppRng eng(seed);

    auto last_work = data.new_known_extractor();
    for (Cluster_ cen = 0; cen < ncenters; ++cen) {
        if (!sofar.empty()) {
            const auto last_ptr = last_work->get_observation(sofar.back());

            parallelize(nthreads, nobs, [&](const int, const Index_ start, const Index_ length) -> void {
                auto curwork = data.new_known_extractor(start, length);
                for (Index_ obs = start, end = start + length; obs < end; ++obs) {
                    const auto current = curwork->get_observation(); // make sure this is before the 'continue', as we MUST call this in every loop iteration to fulfill consecutive access.

                    if (mindist[obs] == 0) {
                        continue;
                    }

                    Float_ r2 = 0;
                    for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
                        const Float_ delta = static_cast<Float_>(current[d]) - static_cast<Float_>(last_ptr[d]); // cast to ensure consistent precision regardless of Data_.
                        r2 += delta * delta;
                    }

                    if (cen == 1 || r2 < mindist[obs]) {
                        mindist[obs] = r2;
                    }
                }
            });
        }

        cumulative[0] = mindist[0];
        for (Index_ i = 1; i < nobs; ++i) {
            cumulative[i] = cumulative[i-1] + mindist[i];
        }

        const auto total = cumulative.back();
        if (total == 0) { // a.k.a. only duplicates left.
            break;
        }

        const auto chosen_id = weighted_sample(cumulative, mindist, nobs, eng);
        mindist[chosen_id] = 0;
        sofar.push_back(chosen_id);
    }

    return sofar;
}

}
/**
 * @endcond
 */

/**
 * @brief **k-means++** initialization of Arthur and Vassilvitskii (2007).
 *
 * Selection of starting points is performed via iterations of weighted sampling, 
 * where the sampling probability for each point is proportional to the squared distance to the closest starting point that was chosen in any of the previous iterations.
 * The aim is to obtain well-separated starting points to encourage the formation of suitable clusters.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * This should be the same as the index type of `Matrix_`.
 * @tparam Data_ Numeric type of the input dataset.
 * This should be the same as the data type of `Matrix_`.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
 *
 * @see
 * Arthur, D. and Vassilvitskii, S. (2007).
 * k-means++: the advantages of careful seeding.
 * _Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
class InitializeKmeanspp final : public Initialize<Index_, Data_, Cluster_, Float_, Matrix_> {
private:
    InitializeKmeansppOptions my_options;

public:
    /**
     * @param options Options for kmeans++ initialization.
     */
    InitializeKmeanspp(InitializeKmeansppOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    InitializeKmeanspp() = default;

public:
    /**
     * @return Options for kmeans++ partitioning.
     * This can be modified prior to calling `run()`.
     */
    InitializeKmeansppOptions& get_options() {
        return my_options;
    }

public:
    /**
     * @cond
     */
    Cluster_ run(const Matrix_& matrix, const Cluster_ ncenters, Float_* const centers) const {
        const auto nobs = matrix.num_observations();
        if (!nobs) {
            return 0;
        }

        const auto sofar = InitializeKmeanspp_internal::run_kmeanspp<Index_, Float_>(matrix, ncenters, my_options.seed, my_options.num_threads);
        internal::copy_into_array(matrix, sofar, centers);
        return sofar.size();
    }
    /**
     * @endcond
     */
};

}

#endif
