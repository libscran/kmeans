#ifndef KMEANS_HARTIGAN_WONG_HPP
#define KMEANS_HARTIGAN_WONG_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

#include "sanisizer/sanisizer.hpp"

#include "Refine.hpp"
#include "Details.hpp"
#include "QuickSearch.hpp"
#include "parallelize.hpp"
#include "compute_centroids.hpp"
#include "is_edge_case.hpp"
#include "utils.hpp"

/**
 * @file RefineHartiganWong.hpp
 *
 * @brief Implements the Hartigan-Wong algorithm for k-means clustering.
 */

namespace kmeans {

/** 
 * @brief Options for `RefineHartiganWong`.
 */
struct RefineHartiganWongOptions {
    /**
     * Maximum number of optimal transfer iterations.
     * More iterations increase the opportunity for convergence at the cost of more compute time.
     */
    int max_iterations = 10;

    /**
     * Maximum number of quick transfer iterations.
     * More iterations increase the opportunity for convergence at the cost of more compute time.
     */
    int max_quick_transfer_iterations = 50;

    /**
     * Whether to quit early when the number of quick transfer iterations exceeds `RefineHartiganWongOptions::max_quick_tranfer_iterations`.
     * Setting this to true recovers the default behavior of R's `kmeans()` implementation.
     * If false, the algorithm will ignore any convergence failures during the quick transfer step and procced to the next optimal transfer iteration.
     * This provides another opportunity to improve the clustering (and potentially achieve convergence) at the cost of more compute time.
     */
    bool quit_on_quick_transfer_convergence_failure = false;

    /** 
     * Number of threads to use.
     * The parallelization scheme is defined by `parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace RefineHartiganWong_internal {

/*
 * Alright, get ready for a data dump, because this is where it gets real.
 * Here I attempt to untangle the code spaghetti from the original Fortran
 * implementation, which is exemplified by the NCP and LIVE arrays.
 *
 * Okay, NCP first. Each element L in NCP has a dual interpretation:
 *
 * - In the optimal-transfer stage, NCP(L) stores the step at which cluster L
 *   was last updated. Each step is just the observation index as the optimal
 *   transfer only does one pass through the dataset.
 * - In the quick-transfer stage, NCP(L) stores the step at which cluster L is
 *   last updated plus M (i.e., the number of observations). Here, the step
 *   corresponding to an observation will be 'M * X + obs' for some integer X
 *   >= 0, where X is the iteration of the quick transfer. This means that the
 *   stored value is defined as '(M + 1) * X + obs'.
 * - After each quick_transfer call, NCP(L) is set back to zero so any existing
 *   values will not carry over into the next optimal_transfer call.
 *
 * Note that these two definitions bleed into each other as the NCP(L) set by
 * optimal_transfer is still referenced in the first few iterations of
 * quick_transfer before it eventually gets overwritten. The easiest way to
 * interpret this is to consider the optimal transfer as iteration 'M = -1'
 * from the perspective of the quick transfer iterations. 
 *
 * In short, the NCP array is used to determine whether a cluster was modified
 * within the last M steps of the optimal_transfer or quick_transfer.
 *
 * Let's move onto LIVE, which has an even more painful interpretation:
 *
 * - Before each optimal_transfer call, LIVE(L) stores the observation at which
 *   cluster L was updated in the _previous_ optimal_transfer call.
 * - During the optimal_transfer call, LIVE(L) is updated to the observation at
 *   which L was updated in this call, plus M (i.e., number of observations).
 * - After the optimal_transfer call, LIVE(L) is updated by subtracting M, so
 *   that the interpretation is correct in the next call.
 * - After the quick_transfer call, LIVE(L) is set to M + 1 if a quick transfer
 *   took place for L, effectively mimicking a "very recent" update.
 *
 * It basically tells us whether there was a recent transfer (optimal or quick)
 * within the last M steps of optimal_transfer. If so, the cluster is "live".
 * This differs very slightly from NCP as LIVE only counts optimal transfer
 * steps while NCP counts both optimal and quick transfer steps.
 *
 * We simplify this mess by consolidating NCP and LIVE into a single update
 * history that can serve both purposes. This is possible as LIVE is not used
 * in quick_transfer, while any modifications made to NCP by quick_transfer are
 * ignored and overwritten in optimal_transfer. Thus, there won't be any
 * conflicts between the two meanings in a consolidated update history.
 *
 * The update history for cluster L now holds the iteration ('X') and the
 * observation ('obs') at which the cluster was last modified. This has
 * an obvious mapping to NCP(L), which was already defined in terms of 
 * 'X' and 'obs' anyway. For LIVE(L), the mapping is more complex:
 *
 * - If an optimal_transfer occurred but no quick_transfer occurs for L,
 *   we set 'X = -2'. 'obs' is not modified as it is the observation at
 *   which L was modified in the previous optimal_transfer iteration;
 *   this is basically just the definition of LIVE(L).
 * - If a quick transfer does occur, we set 'X = -2' and 'obs = M + 1'.
 *   Again, this is equivalent to the behavior of LIVE(L).
 * - If neither a quick_transfer nor optimal_transfer occurs for L,
 *   we set 'X = -3' to indicate that L didn't change in any manner.
 * - In optimal_transfer, we consider a cluster to be live at a particular
 *   observation if its index is greater than 'obs' and its 'X' is -2.
 *
 * Note that Fortran is 1-based so the actual code is a little different than
 * described above for 'obs' - specifically, we just need to set it to 'M'
 * when a quick transfer occurs. Meaning of 'X' is unchanged.
 *
 * Incidentally, we can easily detect if a quick transfer occurs based on
 * whether 'X > -1'. This obviates the need for the ITRAN array.
 */
template<typename Index_>
class UpdateHistory {
private:
    Index_ my_last_observation = 0;

    static constexpr int current_optimal_transfer = -1;
    static constexpr int previous_optimal_transfer = -2;
    static constexpr int ancient_history = -3;

    // Starting at -3 as we can't possibly have any updates if we haven't done
    // any transfers, optimal or quick!
    int my_last_iteration = ancient_history;

public:
    void reset(const Index_ total_obs) {
        if (my_last_iteration > current_optimal_transfer) { // i.e., quick_transfer.
            my_last_observation = total_obs;
            my_last_iteration = previous_optimal_transfer;
        } else if (my_last_iteration == current_optimal_transfer) {
            // Preserve the existing 'my_last_observation', just bump the iteration downwards.
            my_last_iteration = previous_optimal_transfer;
        } else {
            my_last_iteration = ancient_history;
        }
    }

    void set_optimal(const Index_ obs) {
        my_last_observation = obs;
        my_last_iteration = current_optimal_transfer;
    }

    // Here, iter should be from '[0, max_quick_transfer_iterations)'.
    void set_quick(const int iter, const Index_ obs) {
        my_last_iteration = iter;
        my_last_observation = obs;
    }

public:
    bool changed_after(const int iter, const Index_ obs) const {
        if (my_last_iteration == iter) {
            return my_last_observation > obs;
        } else {
            return my_last_iteration > iter;
        }
    }

    bool changed_after_or_at(const int iter, const Index_ obs) const {
        if (my_last_iteration == iter) {
            return my_last_observation >= obs;
        } else {
            return my_last_iteration > iter;
        }
    }

    bool is_live(const Index_ obs) const {
        return changed_after(previous_optimal_transfer, obs);
    }
};

template<typename Float_, typename Index_, typename Cluster_>
struct Workspace {
    // Array arguments in the same order as supplied to R's kmns_ function.
    std::vector<Cluster_> best_destination_cluster; // i.e., IC2
    std::vector<Index_> cluster_sizes; // i.e., NC

    std::vector<Float_> loss_multiplier; // i.e., AN1
    std::vector<Float_> gain_multiplier; // i.e., AN2
    std::vector<Float_> wcss_loss; // i.e., D

    std::vector<UpdateHistory<Index_> > update_history; // i.e., NCP, LIVE, and ITRAN. 

    Index_ optra_steps_since_last_transfer = 0; // i.e., INDX

public:
    Workspace(Index_ nobs, Cluster_ ncenters) :
        // Sizes taken from the .Fortran() call in stats::kmeans(). 
        best_destination_cluster(sanisizer::cast<I<decltype(best_destination_cluster.size())> >(nobs)), 
        cluster_sizes(sanisizer::cast<I<decltype(cluster_sizes.size())> >(ncenters)),
        loss_multiplier(sanisizer::cast<I<decltype(loss_multiplier.size())> >(ncenters)),
        gain_multiplier(sanisizer::cast<I<decltype(gain_multiplier.size())> >(ncenters)),
        wcss_loss(sanisizer::cast<I<decltype(wcss_loss.size())> >(nobs)),
        update_history(sanisizer::cast<I<decltype(update_history.size())> >(ncenters))
    {}
};

template<typename Data_, typename Float_>
Float_ squared_distance_from_cluster(const Data_* const data, const Float_* const center, const std::size_t ndim) {
    Float_ output = 0;
    for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
        const Float_ delta = static_cast<Float_>(data[d]) - center[d]; // cast to float for consistent precision regardless of Data_.
        output += delta * delta;
    }
    return output;
}

template<class Matrix_, typename Cluster_, typename Float_>
void find_closest_two_centers(
    const Matrix_& data,
    const Cluster_ ncenters,
    const Float_* const centers,
    Cluster_* const best_cluster,
    std::vector<Cluster_>& best_destination_cluster,
    const int nthreads)
{
    const auto ndim = data.num_dimensions();

    // We assume that there are at least two centers here, otherwise we should
    // have detected that this was an edge case in RefineHartiganWong::run.
    const internal::QuickSearch<Float_, Cluster_> index(ndim, ncenters, centers);

    const auto nobs = data.num_observations();
    parallelize(nthreads, nobs, [&](const int, const I<decltype(nobs)> start, const I<decltype(nobs)> length) -> void {
        auto matwork = data.new_known_extractor(start, length);
        for (I<decltype(start)> obs = start, end = start + length; obs < end; ++obs) {
            const auto optr = matwork->get_observation();
            const auto res2 = index.find2(optr);
            best_cluster[obs] = res2.first;
            best_destination_cluster[obs] = res2.second;
        }
    });
}

template<typename Float_>
constexpr Float_ big_number() {
    return 1e30; // Some very big number.
}

template<typename Data_, typename Index_, typename Cluster_, typename Float_>
void transfer_point(
    const std::size_t ndim,
    const Data_* const obs_ptr,
    const Index_ obs_id,
    const Cluster_ l1,
    const Cluster_ l2,
    Float_* const centers,
    Cluster_* const best_cluster,
    Workspace<Float_, Index_, Cluster_>& work)
{
    // Yes, casts to float are deliberate here, so that the
    // multipliers can be computed correctly.
    const Float_ al1 = work.cluster_sizes[l1], alw = al1 - 1;
    const Float_ al2 = work.cluster_sizes[l2], alt = al2 + 1;

    for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
        const Float_ oval = obs_ptr[d]; // cast to float for consistent precision regardless of Data_.
        auto& c1 = centers[sanisizer::nd_offset<std::size_t>(d, ndim, l1)];
        c1 = (c1 * al1 - oval) / alw;
        auto& c2 = centers[sanisizer::nd_offset<std::size_t>(d, ndim, l2)];
        c2 = (c2 * al2 + oval) / alt;
    }

    --work.cluster_sizes[l1];
    ++work.cluster_sizes[l2];

    work.gain_multiplier[l1] = alw / al1;
    work.loss_multiplier[l1] = (alw > 1 ? alw / (alw - 1) : big_number<Float_>());
    work.loss_multiplier[l2] = alt / al2;
    work.gain_multiplier[l2] = alt / (alt + 1);

    best_cluster[obs_id] = l2;
    work.best_destination_cluster[obs_id] = l1;
}

/* ALGORITHM AS 136.1  APPL. STATIST. (1979) VOL.28, NO.1
 * This is the OPtimal TRAnsfer stage.
 *             ----------------------
 * Each point is re-assigned, if necessary, to the cluster that will induce a
 * maximum reduction in the within-cluster sum of squares. In this stage,
 * there is only one pass through the data.
 */
template<class Matrix_, typename Cluster_, typename Float_>
bool optimal_transfer(
    const Matrix_& data, Workspace<Float_, Index<Matrix_>, Cluster_>& work,
    const Cluster_ ncenters,
    Float_* const centers,
    Cluster_* const best_cluster,
    const bool all_live)
{
    const auto nobs = data.num_observations();
    const auto ndim = data.num_dimensions();
    auto matwork = data.new_known_extractor();

    for (I<decltype(nobs)> obs = 0; obs < nobs; ++obs) { 
        ++work.optra_steps_since_last_transfer;

        const auto l1 = best_cluster[obs];
        if (work.cluster_sizes[l1] != 1) {
            const auto obs_ptr = matwork->get_observation(obs);

            // The original Fortran implementation re-used the WCSS loss for
            // each observation, only recomputing it if its cluster had
            // experienced an optimal transfer for an earlier observation. In
            // theory, this sounds great to avoid recomputation, but the
            // existing WCSS loss was computed in a running fashion during the
            // quick transfers. This makes them susceptible to accumulation of
            // numerical errors, even after the centroids are freshly
            // recomputed in the run() loop. So, we simplify matters and
            // improve accuracy by just recomputing the loss all the time,
            // which doesn't take too much extra effort.
            auto& wcss_loss = work.wcss_loss[obs];
            const auto l1_ptr = centers + sanisizer::product_unsafe<std::size_t>(ndim, l1);
            wcss_loss = squared_distance_from_cluster(obs_ptr, l1_ptr, ndim) * work.loss_multiplier[l1];

            // Find the cluster with minimum WCSS gain.
            auto l2 = work.best_destination_cluster[obs];
            const auto original_l2 = l2;
            const auto l2_ptr = centers + sanisizer::product_unsafe<std::size_t>(ndim, l2);
            auto wcss_gain = squared_distance_from_cluster(obs_ptr, l2_ptr, ndim) * work.gain_multiplier[l2];

            const auto update_destination_cluster = [&](const Cluster_ cen) -> void {
                auto cen_ptr = centers + sanisizer::product_unsafe<std::size_t>(ndim, cen);
                auto candidate = squared_distance_from_cluster(obs_ptr, cen_ptr, ndim) * work.gain_multiplier[cen];
                if (candidate < wcss_gain) {
                    wcss_gain = candidate;
                    l2 = cen;
                }
            };

            // If the current assigned cluster is live, we need to check all
            // other clusters as potential transfer destinations, because the
            // gain/loss comparison has changed. Otherwise, we only need to
            // consider other live clusters as transfer destinations; the
            // non-live clusters were already rejected as transfer destinations
            // when compared to the current assigned cluster, so there's no
            // point checking them again if they didn't change in the meantime.
            //
            // The exception is for the first call to optimal_transfer, where we
            // consider all clusters as live (i.e., all_live = true). This is
            // because no observation really knows its best transfer
            // destination yet - the second-closest cluster is just a
            // guesstimate - so we need to compute it exhaustively.
            if (all_live || work.update_history[l1].is_live(obs)) { 
                for (Cluster_ cen = 0; cen < ncenters; ++cen) {
                    if (cen != l1 && cen != original_l2) {
                        update_destination_cluster(cen);
                    }
                }
            } else {
                for (Cluster_ cen = 0; cen < ncenters; ++cen) {
                    if (cen != l1 && cen != original_l2 && work.update_history[cen].is_live(obs)) {
                        update_destination_cluster(cen);
                    }
                }
            }

            // Deciding whether to make the transfer based on the change to the WCSS.
            if (wcss_gain >= wcss_loss) {
                work.best_destination_cluster[obs] = l2;
            } else {
                work.optra_steps_since_last_transfer = 0;
                work.update_history[l1].set_optimal(obs);
                work.update_history[l2].set_optimal(obs);
                transfer_point(ndim, obs_ptr, obs, l1, l2, centers, best_cluster, work);
            }
        }

        // Stop if we've iterated through the entire dataset and no transfer of
        // any kind took place, be it optimal or quick.
        if (work.optra_steps_since_last_transfer == nobs) {
            return true;
        }
    }

    return false;
} 

/* ALGORITHM AS 136.2  APPL. STATIST. (1979) VOL.28, NO.1 
 * This is the Quick TRANsfer stage. 
 *             -------------------- 
 * IC1(I) is the cluster which point I currently belongs to.
 * IC2(I) is the cluster which point I is most likely to be transferred to.
 *
 * For each point I, IC1(I) & IC2(I) are switched, if necessary, to reduce
 * within-cluster sum of squares. The cluster centres are updated after each
 * step. In this stage, we loop through the data until no further change is to
 * take place, or we hit an iteration limit, whichever is first.
 */
template<class Matrix_, typename Cluster_, typename Float_>
std::pair<bool, bool> quick_transfer(
    const Matrix_& data,
    Workspace<Float_, Index<Matrix_>, Cluster_>& work,
    Float_* const centers,
    Cluster_* const best_cluster,
    const int quick_iterations)
{
    bool had_transfer = false;

    const auto nobs = data.num_observations();
    const auto ndim = data.num_dimensions();
    auto matwork = data.new_known_extractor();

    I<decltype(nobs)> steps_since_last_quick_transfer = 0; // i.e., ICOUN in the original Fortran implementation.

    for (int it = 0; it < quick_iterations; ++it) {
        const int prev_it = it - 1;

        for (I<decltype(nobs)> obs = 0; obs < nobs; ++obs) { 
            ++steps_since_last_quick_transfer;
            const auto l1 = best_cluster[obs];

            if (work.cluster_sizes[l1] != 1) {
                I<decltype(matwork->get_observation(obs))> obs_ptr = NULL;

                // Need to update the WCSS loss if the cluster was updated recently. 
                // Otherwise, we must have already updated the WCSS in a previous 
                // iteration of the outermost loop, so this can be skipped.
                //
                // Note that we use changed_at_or_after; if the same
                // observation was changed in the previous iteration of the
                // outermost loop, its WCSS loss won't have been updated yet.
                auto& history1 = work.update_history[l1];
                if (history1.changed_after_or_at(prev_it, obs)) {
                    const auto l1_ptr = centers + sanisizer::product_unsafe<std::size_t>(l1, ndim);
                    obs_ptr = matwork->get_observation(obs);
                    work.wcss_loss[obs] = squared_distance_from_cluster(obs_ptr, l1_ptr, ndim) * work.loss_multiplier[l1];
                }

                // If neither the best or second-best clusters have changed
                // after the previous iteration that we visited this
                // observation, then there's no point reevaluating the
                // transfer, because nothing's going to be different anyway.
                const auto l2 = work.best_destination_cluster[obs];
                auto& history2 = work.update_history[l2];
                if (history1.changed_after(prev_it, obs) || history2.changed_after(prev_it, obs)) {
                    if (obs_ptr == NULL) {
                        obs_ptr = matwork->get_observation(obs);
                    }
                    const auto l2_ptr = centers + sanisizer::product_unsafe<std::size_t>(l2, ndim);
                    const auto wcss_gain = squared_distance_from_cluster(obs_ptr, l2_ptr, ndim) * work.gain_multiplier[l2];

                    if (wcss_gain < work.wcss_loss[obs]) {
                        had_transfer = true;
                        steps_since_last_quick_transfer = 0;
                        history1.set_quick(it, obs);
                        history2.set_quick(it, obs);
                        transfer_point(ndim, obs_ptr, obs, l1, l2, centers, best_cluster, work);
                    }
               }
           }

           if (steps_since_last_quick_transfer == nobs) {
               // Quit early if no transfer occurred within the past 'nobs'
               // steps, as we've already converged for each observation. 
               return std::make_pair(had_transfer, false);
           }
        }
    } 

    return std::make_pair(had_transfer, true);
}

}
/**
 * @endcond
 */

/**
 * @brief Implements the Hartigan-Wong algorithm for k-means clustering.
 *
 * The Hartigan-Wong algorithm performs several iterations of transferring points between clusters, 
 * involving a computationally expensive "optimal transfer" step that checks each observation against each cluster to determine its best assignment,
 * followed by a cheaper "quick transfer" step, which iterates between the best and second-best cluster choices for each observation.
 * The latter enables rapid exploration of the local solution space without the unnecessary expense of repeatedly comparing each observation to all clusters.
 * The choice of "best" cluster for each observation considers the gain/loss in the sum of squares when an observation moves between clusters,
 * even accounting for the shift in the cluster centers after the transfer.
 * The algorithm terminates when no observation wishes to transfer between clusters.
 *
 * This implementation is derived from the Fortran code underlying the `kmeans` function in the **stats** R package,
 * which in turn is taken from Hartigan and Wong (1979).
 * 
 * In the `Details::status` returned by `run()`, the status code is either 0 (success),
 * 2 (maximum optimal transfer iterations reached without convergence)
 * or 4 (maximum quick transfer iterations reached without convergence, if `RefineHartiganWongOptions::quit_on_quick_transfer_convergence_failure = true`).
 * Previous versions of the library would report a status code of 1 upon encountering an empty cluster, but these are now just ignored.
 *
 * In the `Details::iterations` returned by `run()`, the reported number of iterations is that of the optimal transfers.
 *
 * @tparam Index_ Integer type of the observation indices. 
 * This should be the same as the index type of `Matrix_`.
 * @tparam Data_ Numeric type of the input dataset.
 * This should be the same as the data type of `Matrix_`.
 * @tparam Cluster_ Integer type of the cluster assignments.
 * @tparam Float_ Floating-point type of the centroids.
 * This will also be used for the internal distance calculations.
 * @tparam Matrix_ Class satisfying the `Matrix` interface.
 *
 * @see
 * Hartigan, J. A. and Wong, M. A. (1979).
 * Algorithm AS 136: A K-means clustering algorithm.
 * _Applied Statistics_, 28, 100-108.
 */
template<typename Index_, typename Data_, typename Cluster_, typename Float_, class Matrix_ = Matrix<Index_, Data_> >
class RefineHartiganWong final : public Refine<Index_, Data_, Cluster_, Float_, Matrix_> {
public:
    /**
     * @param options Further options for the Hartigan-Wong algorithm.
     */
    RefineHartiganWong(RefineHartiganWongOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor. 
     */
    RefineHartiganWong() = default;

private:
    RefineHartiganWongOptions my_options;

public:
    /**
     * @return Options for Hartigan-Wong clustering.
     * This can be modified prior to calling `run()`.
     */
    RefineHartiganWongOptions& get_options() {
        return my_options;
    }

public:
    /**
     * @cond
     */
    Details<Index_> run(const Matrix_& data, const Cluster_ ncenters, Float_* const centers, Cluster_* const clusters) const {
        const auto nobs = data.num_observations();
        if (internal::is_edge_case(nobs, ncenters)) {
            return internal::process_edge_case(data, ncenters, centers, clusters);
        }

        RefineHartiganWong_internal::Workspace<Float_, Index_, Cluster_> work(nobs, ncenters);

        RefineHartiganWong_internal::find_closest_two_centers(data, ncenters, centers, clusters, work.best_destination_cluster, my_options.num_threads);
        for (Index_ obs = 0; obs < nobs; ++obs) {
            ++work.cluster_sizes[clusters[obs]];
        }
        internal::compute_centroids(data, ncenters, centers, clusters, work.cluster_sizes);

        for (Cluster_ cen = 0; cen < ncenters; ++cen) {
            const Float_ num = work.cluster_sizes[cen]; // yes, cast is deliberate here so that the multipliers can be computed correctly.
            work.gain_multiplier[cen] = num / (num + 1);
            work.loss_multiplier[cen] = (num > 1 ? num / (num - 1) : RefineHartiganWong_internal::big_number<Float_>());
        }

        I<decltype(my_options.max_iterations)> iter = 0;
        int ifault = 0;
        for (; iter < my_options.max_iterations; ++iter) {
            const bool finished = RefineHartiganWong_internal::optimal_transfer(data, work, ncenters, centers, clusters, /* all_live = */ (iter == 0));
            if (finished) {
                break;
            }

            const auto quick_status = RefineHartiganWong_internal::quick_transfer(
                data,
                work,
                centers,
                clusters,
                my_options.max_quick_transfer_iterations
            );

            // Recomputing the centroids to avoid accumulation of numerical
            // errors after many transfers (e.g., adding a whole bunch of
            // values and then subtracting them again leaves behind some
            // cancellation error). Note that we don't have to do this if
            // 'finished = true' as this means that there was no transfer of
            // any kind in the final pass through the dataset.
            internal::compute_centroids(data, ncenters, centers, clusters, work.cluster_sizes);

            if (quick_status.second) { // Hit the quick transfer iteration limit.
                if (my_options.quit_on_quick_transfer_convergence_failure) {
                    ifault = 4;
                    break;
                }
            } else {
                // If quick transfer converged and there are only two clusters,
                // there is no need to re-enter the optimal transfer stage. 
                if (ncenters == 2) {
                    break;
                }
            }

            if (quick_status.first) { // At least one quick transfer was performed.
                work.optra_steps_since_last_transfer = 0;
            }

            for (Cluster_ c = 0; c < ncenters; ++c) {
                work.update_history[c].reset(nobs);
            }
        }

        if (iter == my_options.max_iterations) {
            ifault = 2;
        } else {
            ++iter; // make it 1-based.
        }

        return Details(std::move(work.cluster_sizes), iter, ifault);
    }
    /**
     * @endcond
     */
};

}

#endif
