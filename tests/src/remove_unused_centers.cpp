#include <gtest/gtest.h>
#include "kmeans/remove_unused_centers.hpp"

TEST(RemoveUnusedCenters, NoOp) {
    const std::size_t num_dim = 5;
    const int num_cen = 5;

    std::vector<double> centers;
    std::vector<int> clusters;
    std::vector<int> sizes;
    for (int k = 0; k < num_cen; ++k) {
        centers.insert(centers.end(), num_dim, k);
        sizes.push_back(k + 5);
        clusters.insert(clusters.end(), sizes.back(), k);
    }

    auto clusters2 = clusters;
    auto centers2 = centers;
    auto sizes2 = sizes;
    const int num_obs = clusters.size();
    const auto out = kmeans::remove_unused_centers(num_dim, num_obs, clusters2.data(), num_cen, centers2.data(), sizes2);

    EXPECT_EQ(out, num_cen);
    EXPECT_EQ(sizes2, sizes);
    EXPECT_EQ(centers2, centers);
    EXPECT_EQ(clusters2, clusters);
}

TEST(RemoveUnusedCenters, Removed) {
    const std::size_t num_dim = 5;
    const int num_cen = 7;

    std::vector<double> centers, true_centers;
    std::vector<int> clusters, true_clusters;
    std::vector<int> sizes, true_sizes;
    for (int k = 0; k < num_cen; ++k) {
        centers.insert(centers.end(), num_dim, k);
        if (k % 2 == 0) {
            sizes.push_back(k + 5);
            clusters.insert(clusters.end(), sizes.back(), k);
            true_clusters.insert(true_clusters.end(), sizes.back(), true_sizes.size());
            true_centers.insert(true_centers.end(), num_dim, k);
            true_sizes.push_back(sizes.back());
        } else {
            sizes.push_back(0);
        }
    }

    true_sizes.resize(sizes.size());
    true_centers.resize(centers.size());

    const int num_obs = clusters.size();
    const auto out = kmeans::remove_unused_centers(num_dim, num_obs, clusters.data(), num_cen, centers.data(), sizes, 0.0);

    EXPECT_EQ(out, 4);
    EXPECT_EQ(sizes, true_sizes);
    EXPECT_EQ(centers, true_centers);
    EXPECT_EQ(clusters, true_clusters);
}

