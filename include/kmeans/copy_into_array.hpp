#ifndef KMEANS_COPY_HPP
#define KMEANS_COPY_HPP

#include <algorithm>
#include <vector>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

namespace kmeans {

namespace internal {

template<class Matrix_, typename Float_>
void copy_into_array(const Matrix_& matrix, const std::vector<Index<Matrix_> >& chosen, Float_* const out) {
    const auto ndim = matrix.num_dimensions();
    const auto nchosen = chosen.size();
    auto work = matrix.new_known_extractor(chosen.data(), sanisizer::cast<std::size_t>(nchosen));
    for (I<decltype(nchosen)> i = 0; i < nchosen; ++i) {
        const auto ptr = work->get_observation();
        std::copy_n(ptr, ndim, out + sanisizer::product_unsafe<std::size_t>(i, ndim));
    }
}

}

}

#endif
