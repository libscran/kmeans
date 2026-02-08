#ifndef KMEANS_QUICKSEARCH_HPP
#define KMEANS_QUICKSEARCH_HPP

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <queue>
#include <cstddef>
#include <type_traits>
#include <array>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

namespace kmeans {

namespace internal {

/************************************************************
 *** Implementation adapted from knncolle::VptreePrebuilt ***
 ************************************************************/

template<typename Float_, typename Index_>
class QuickSearch {
private:
    std::size_t my_dim;
    Index_ my_pts;

    struct BuildHistory {
        BuildHistory(Index_ lower, Index_ upper, Index_* right) : right(right), lower(lower), upper(upper) {}
        Index_* right; // This is a pointer to the 'Node::right' of the parent of the node-to-be-added.
        Index_ lower, upper; // Lower and upper ranges of the items in the node-to-be-added.
    };

    typedef std::pair<Float_, Index_> DataPoint; 
    std::vector<DataPoint> my_items;
    std::vector<BuildHistory> my_history;
    std::mt19937_64 my_rng;

    // Normally, 'left' or 'right' must be > 0, as the first node in 'nodes' is
    // the root and cannot be referenced from other nodes. This means that we
    // can use 0 as a sentinel to indicate that no child exists here.
    static constexpr Index_ TERMINAL = 0;

    // Single node of a VP tree. 
    struct Node {
        Float_ radius = 0;

        // Original index of current vantage point, defining the center of the node.
        Index_ index = 0;

        // Node index of the next vantage point for all children no more than 'threshold' from the current vantage point.
        Index_ left = TERMINAL;

        // Node index of the next vantage point for all children no less than 'threshold' from the current vantage point.
        Index_ right = TERMINAL; 
    };

    std::vector<Node> my_nodes;

    std::vector<Float_> my_data;

public:
    template<typename Query_>
    static Float_ raw_distance(const Float_* const x, const Query_* const y, const std::size_t ndim) {
        Float_ output = 0;
        for (I<decltype(ndim)> d = 0; d < ndim; ++d) {
            const Float_ delta = x[d] - static_cast<Float_>(y[d]); // cast to ensure consistent precision regardless of Query_.
            output += delta * delta;
        }
        return output;
    }

    void build(const Float_* coords) {
        sanisizer::reserve(my_items, my_pts);
        my_items.clear();
        for (Index_ i = 0; i < my_pts; ++i) {
            my_items.emplace_back(0, i);
        }

        // Nodes are already reserved ahead of time to avoid allocations that could invalidate pointers.
        sanisizer::reserve(my_nodes, my_pts);
        my_nodes.clear();

        // We're assuming that lower < upper at each loop. This requires some protection at the call site when nobs = 0, see the constructor.
        my_history.clear();
        Index_ lower = 0, upper = my_pts;
        while (1) {
            my_nodes.emplace_back();
            Node& node = my_nodes.back(); 

            const Index_ gap = upper - lower;
            assert(gap > 0);
            if (gap == 1) { // i.e., we're at a leaf.
                const auto& leaf = my_items[lower];
                node.index = leaf.second;

                // If we're at a leaf, we've finished this particular branch of the tree, so we can start rolling back through history.
                if (my_history.empty()) {
                    return;
                }
                *(my_history.back().right) = my_nodes.size();
                lower = my_history.back().lower;
                upper = my_history.back().upper;
                my_history.pop_back();
                continue;
            }

            // Choose an arbitrary point and move it to the start of the [lower, upper) interval in 'my_items'; this is our new vantage point.
            // Yes, I know that the modulo method does not provide strictly uniform values but statistical correctness doesn't really matter here,
            // and I don't want std::uniform_int_distribution's implementation-specific behavior.
            const Index_ vp = (my_rng() % gap + lower);
            std::swap(my_items[lower], my_items[vp]);
            const auto& vantage = my_items[lower];
            node.index = vantage.second;
            const auto vantage_ptr = coords + sanisizer::product_unsafe<std::size_t>(vantage.second, my_dim);

            // Compute distances to the new vantage point.
            // We +1 to exclude the vantage point itself, obviously.
            const Index_ lower_p1 = lower + 1;
            for (Index_ i = lower_p1 ; i < upper; ++i) {
                const auto loc = coords + sanisizer::product_unsafe<std::size_t>(my_items[i].second, my_dim);
                my_items[i].first = raw_distance(vantage_ptr, loc, my_dim);
            }

            if (gap > 2) {
                // Partition around the median distance from the vantage point.
                const Index_ median = lower_p1 + (gap - 1)/2;
                std::nth_element(my_items.begin() + lower_p1, my_items.begin() + median, my_items.begin() + upper);

                // Radius of the new node will be the distance to the median.
                node.radius = std::sqrt(my_items[median].first);

                // The next iteration will process the left node (i.e., inside the ball).
                // We store the boundaries of the yet-to-be-added right node to the history for later processing.
                my_history.emplace_back(median, upper, &(node.right));
                node.left = my_nodes.size();
                lower = lower_p1;
                upper = median;

            } else {
                // Here we only have one child, as this node has two points and one of them was already used as the vantage point.
                // So the other point is used directly as the right node.
                const Index_ median = lower_p1;
                node.radius = std::sqrt(my_items[median].first);
                node.right = my_nodes.size();
                lower = median;

                // Several points worth mentioning here:
                // - No need to set upper, as we'd end up just doing upper = upper and clang complains.
                // - This code allows us to get a node where left = TERMINAL and right != TERMINAL, but the opposite is impossible.
                //   This fact is exploited in search_best() for some minor optimizations.
            }
        }
    }

public:
    QuickSearch(const std::size_t num_dim, const Index_ num_pts) : 
        my_dim(num_dim),
        my_pts(num_pts),
        my_rng([&]() {
            // Statistical correctness doesn't matter (aside from tie breaking)
            // so we'll just use a deterministically 'random' number to ensure
            // we get the same ties for any given dataset but a different stream
            // of numbers between datasets. Casting to get well-defined overflow.
            typedef std::mt19937_64 Engine;
            const typename std::make_unsigned<typename Engine::result_type>::type base = 1234567890, m1 = my_pts, m2 = my_dim;
            return base * m1 +  m2;
        }()),
        my_data(sanisizer::product<I<decltype(my_data.size())> >(my_dim, my_pts))
    {
    }

    void reset(const Float_* const data) {
        if (my_pts == 0) {
            return;
        }

        build(data);

        // Copying the data over in order of occurence in the tree, for more cache-friendliness.
        for (Index_ i = 0; i < my_pts; ++i) {
            std::copy_n(
                data + sanisizer::product_unsafe<std::size_t>(my_nodes[i].index, my_dim),
                my_dim,
                my_data.data() + sanisizer::product_unsafe<std::size_t>(i, my_dim)
            );
        }
    }

public:
    struct Workspace {
        std::vector<std::pair<bool, Index_> > history;
    };

    auto new_workspace() const {
        return Workspace();
    }

private:
    static bool can_progress_left(const Node& node, const Float_ dist_to_vp, const Float_ threshold) {
        return node.left != TERMINAL && dist_to_vp - threshold <= node.radius; 
    }

    static bool can_progress_right(const Node& node, const Float_ dist_to_vp, const Float_ threshold) {
        // Using >= in the triangle inequality as there are some points that lie on the surface of the ball but are considered 'outside' the ball,
        // e.g., the median point itself as well as anything with a tied distance.
        return node.right != TERMINAL && dist_to_vp + threshold >= node.radius; 
    }

    template<typename Query_, class Store_> 
    void search_best(const Query_* query, Store_ store, Workspace& work) const { 
        work.history.clear();
        Index_ curnode_offset = 0;
        Float_ max_dist = std::numeric_limits<Float_>::max();

        while (1) {
            auto nptr = my_data.data() + sanisizer::product_unsafe<std::size_t>(curnode_offset, my_dim);
            const Float_ dist_to_vp = std::sqrt(raw_distance(nptr, query, my_dim));

            const auto& curnode = my_nodes[curnode_offset];
            if (dist_to_vp <= max_dist) {
                max_dist = store(curnode.index, dist_to_vp);
            }

            if (dist_to_vp < curnode.radius) {
                // If the target lies within the radius of ball, chances are that its neighbors also lie inside the ball.
                // So we check the points inside the ball first (i.e., left node) to try to shrink max_dist as fast as possible.

                // A quirk here is that, if dist_to_vp < curnode.radius, then can_progress_left must be true if curnode.left != TERMINAL.
                // So we don't bother to compute the full function.
                const bool can_left = curnode.left != TERMINAL;
                const bool can_right = can_progress_right(curnode, dist_to_vp, max_dist);

                if (can_left) {
                    if (can_right) {
                        work.history.emplace_back(false, curnode_offset);
                    }
                    curnode_offset = curnode.left;
                    continue;
                } else if (can_right) {
                    curnode_offset = curnode.right;
                    continue;
                }

            } else {
                // Otherwise, if the target lies at or outside the radius of the ball, chances are its neighbors also lie outside the ball.
                // So we check the points outside the ball first (i.e., right node) to try to shrink max_dist as fast as possible.

                // A quirk here is that, if dist_to_vp >= curnode.radius, then can_progress_right must be true if curnode.right != TERMINAL.
                // So we don't bother to compute the full function.
                const bool can_right = curnode.right != TERMINAL;
                const bool can_left = can_progress_left(curnode, dist_to_vp, max_dist);

                if (can_right) {
                    if (can_left) {
                        work.history.emplace_back(true, curnode_offset);
                    }
                    curnode_offset = curnode.right;
                    continue;
                } else {
                    // The manner of construction of the VP tree prevents the existence of a node where right == TERMINAL but left != TERMINAL.
                    // As such, there's no need to consider the 'else if (can_left) {' condition that we would otherwise expect for symmetry with the inside-ball code.
                    assert(!can_left);
                }
            }

            // We don't have anything else to do here, so we move back to the last branching node in our history. 
            if (work.history.empty()) {
                return;
            }

            auto& histinfo = work.history.back(); 
            if (!histinfo.first) {
                curnode_offset = my_nodes[histinfo.second].right; 
            } else {
                curnode_offset = my_nodes[histinfo.second].left;
            }
            work.history.pop_back();
        }
    }

public:
    template<bool test_ = false, typename Query_>
    auto find(const Query_* const query, Workspace& work) const {
        std::pair<Float_, Index_> closest{ std::numeric_limits<Float_>::max(), std::numeric_limits<Index_>::max() };
        search_best(
            query, 
            [&](Index_ index, Float_ dist) -> Float_ {
                // We use min() on pairs here to handle ties.
                // Specifically, on tied distances, we favor the point with the lower 'index'.
                std::pair<Float_, Index_> candidate { dist, index };
                if (candidate < closest) {
                    closest = candidate;
                }
                return dist;
            },
            work
        );

        if constexpr(test_) {
            return std::pair<Index_, Float_>{ closest.second, closest.first };
        } else {
            return closest.second;
        }
    }

    template<typename Query_>
    std::array<Index_, 2> find2(const Query_* query, Workspace& work) const {
        // Storing our history as pairs so that comparison will consider the index in the presence of tied distances. 
        std::pair<Float_, Index_> closest{ std::numeric_limits<Float_>::max(), std::numeric_limits<Index_>::max() };
        std::pair<Float_, Index_> second_closest{ std::numeric_limits<Float_>::max(), std::numeric_limits<Index_>::max() };

        search_best(
            query,
            [&](Index_ index, Float_ dist) -> Float_ {
                std::pair<Float_, Index_> candidate { dist, index };
                if (candidate < closest) {
                    second_closest = closest;
                    closest = candidate;
                    return second_closest.first;
                } else {
                    if (candidate < second_closest) {
                        second_closest = candidate;
                    }
                    return dist;
                }
            },
            work
        );

        // Better be more than two points, otherwise the second is set to the placeholder value!
        return std::array<Index_, 2>{ closest.second, second_closest.second };
    }
};

}

}

#endif
