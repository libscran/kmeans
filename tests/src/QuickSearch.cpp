#include "TestCore.h"

#ifdef CUSTOM_PARALLEL_TEST
// Must be before any kmeans imports.
#include "custom_parallel.h"
#endif

#include "kmeans/QuickSearch.hpp"

class QuickSearchTest : public TestCore, public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    void SetUp() {
        assemble(GetParam());
    }
};

TEST_P(QuickSearchTest, Sweep) {
    // Quick and dirty check to verify that we do get back the right identity.
    {
        kmeans::internal::QuickSearch<double, int> index(nr, nc);
        index.reset(data.data()); 
        auto wrk = index.new_workspace();
        for (int c = 0; c < nc; ++c) {
            auto best = index.find(data.data() + c * nr, wrk);
            EXPECT_EQ(c, best);
        }

        // Still works when we reset it.
        auto copy = data;
        for (auto& x : copy) {
            x /= 2;
        }
        index.reset(copy.data());
        for (int c = 0; c < nc; ++c) {
            auto best = index.find(copy.data() + c * nr, wrk);
            EXPECT_EQ(c, best);
        }
    }

    // A more serious test with non-identical inputs.
    {
        auto half_nc = nc/2;
        kmeans::internal::QuickSearch<double, int> half_index(nr, half_nc);
        half_index.reset(data.data()); 
        auto work = half_index.new_workspace();
        std::vector<std::pair<int, double> > collected(nc);

        for (int c = half_nc; c < nc; ++c) {
            auto self = data.data() + c * nr;
            auto best = half_index.template find<true>(self, work);

            double expected_dist = std::numeric_limits<double>::infinity();
            int expected_best = 0;
            for (int b = 0; b < half_nc; ++b) {
                double d2 = 0;
                auto other = data.data() + b * nr;
                for (int r = 0; r < nr; ++r) {
                    double delta = other[r] - self[r];
                    d2 += delta * delta;
                }
                if (d2 < expected_dist) {
                    expected_best = b;
                    expected_dist = d2;
                }
            }

            EXPECT_EQ(expected_best, best.first);
            EXPECT_EQ(std::sqrt(expected_dist), best.second);
            collected[c] = best;
        }

        // Still works when we reset it.
        half_index.reset(data.data()); 
        for (int c = half_nc; c < nc; ++c) {
            auto self = data.data() + c * nr;
            auto best = half_index.template find<true>(self, work);
            EXPECT_EQ(collected[c], best);
        }
    }
}

TEST_P(QuickSearchTest, TakeTwo) {
    auto param = GetParam();
    assemble(param);

    kmeans::internal::QuickSearch<double, int> index(nr, nc);
    index.reset(data.data());
    auto work = index.new_workspace();    

    for (int c = 0; c < nc; ++c) {
        auto res = index.find2(data.data() + c * nr, work);
        EXPECT_EQ(c, res[0]);

        auto self = data.data() + c * nr;
        double expected_dist = std::numeric_limits<double>::infinity();
        int expected_second = 0;
        for (int b = 0; b < nc; ++b) {
            if (b == c) {
                continue;
            }

            double d2 = 0;
            auto other = data.data() + b * nr;
            for (int r = 0; r < nr; ++r) {
                double delta = other[r] - self[r];
                d2 += delta * delta;
            }

            if (d2 < expected_dist) {
                expected_second = b;
                expected_dist = d2;
            }
        }

        EXPECT_EQ(expected_second, res[1]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuickSearch,
    QuickSearchTest,
    ::testing::Combine(
        ::testing::Values(10, 20), // number of dimensions
        ::testing::Values(2, 10, 50) // number of observations 
    )
);

TEST(QuickSearch, TieBreaker) {
    const int nr = 10, nc = 20;
    std::vector<double> data(nr * nc);
    std::fill(data.begin() + nr * (nc / 2), data.end(), 1);

    kmeans::internal::QuickSearch<double, int> index(nr, nc);
    index.reset(data.data());
    auto work = index.new_workspace();    

    {
        auto first = data.data();
        EXPECT_EQ(index.find(first, work), 0);
        auto pair = index.find2(first, work);
        EXPECT_EQ(pair[0], 0);
        EXPECT_EQ(pair[1], 1);
    }

    {
        auto last = data.data() + data.size() - nr;
        auto work = index.new_workspace();    
        EXPECT_EQ(index.find(last, work), 10);
        auto pair = index.find2(last, work);
        EXPECT_EQ(pair[0], 10);
        EXPECT_EQ(pair[1], 11);
    }

    // Checking that it still behaves upon reset.
    std::reverse(data.begin(), data.end());
    index.reset(data.data());

    {
        auto first = data.data();
        EXPECT_EQ(index.find(first, work), 0);
        auto pair = index.find2(first, work);
        EXPECT_EQ(pair[0], 0);
        EXPECT_EQ(pair[1], 1);
    }

    {
        auto last = data.data() + data.size() - nr;
        auto work = index.new_workspace();    
        EXPECT_EQ(index.find(last, work), 10);
        auto pair = index.find2(last, work);
        EXPECT_EQ(pair[0], 10);
        EXPECT_EQ(pair[1], 11);
    }
}
