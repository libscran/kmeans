#ifndef KMEANS_SIMPLE_MATRIX_HPP
#define KMEANS_SIMPLE_MATRIX_HPP

#include <cstddef>
#include <limits>

#include "sanisizer/sanisizer.hpp"

#include "Matrix.hpp"

/**
 * @file SimpleMatrix.hpp
 * @brief Wrapper for a simple dense matrix.
 */

namespace kmeans {

/**
 * @cond
 */
template<typename Index_, typename Data_>
class SimpleMatrix;

#ifdef KMEANS_DEBUG_POINTER_USAGE
/**
 * When testing, we want to check that the pointers returned by
 * get_observation() are used correctly in other **kmeans** library functions.
 * Specifically, the pointer returned by get_observation() should not be used
 * after the next get_observation() call.
 *
 * Unfortunately, it's hard to confirm that this is really the case with the
 * usual SimpleMatrix, because the pointers just refer to a contiguous array
 * and will always point to the correct location for the lifetime of the
 * SimpleMatrix. To overcome this, when KMEANS_DEBUG_POINTER_USAGE is defined,
 * we... mix things up a bit by copying the data to a smaller intermediate
 * array, overwriting any previous contents. We then return a pointer to this
 * intermediate array; if that pointer is used after the next get_observation()
 * call, it will now refer to the wrong contents and break some tests.
 *
 * To make it a bit more obvious that something has gone wrong, we fill the
 * intermediate array with NaNs and we randomly select part of the array to
 * copy to. Thus, if a pointer is re-used, there is a chance that it just
 * refers to a bunch of NaNs, which should definitely break some tests.
 *
 * TBH, it might be less error-prone to follow the tatami model of forcing the
 * caller to supply their own memory to get_observation(), which would make it
 * clear to the caller that different allocations are required for multiple
 * calls. But this would force an unnecessary allocation for the most common
 * SimpleMatrix use case, and besides, if the caller accidentally re-used
 * allocations for different calls, we'd be back at the same problem. (Less of
 * an issue for tatami itself as downstream functions are always tested on 
 * different matrix representations, but here, we've only got a SimpleMatrix.)
 */

template<typename Data_>
class Buffer {
public:
    Buffer(std::size_t n) : my_buffer(n * MULT), my_n(n), my_eng(n + 1) {}

private:
    static constexpr std::size_t MULT = 5;
    std::vector<Data_> my_buffer;
    std::size_t my_n;
    std::mt19937_64 my_eng;

public:
    const Data_* update(const Data_* const ptr) {
        std::fill(my_buffer.begin(), my_buffer.end(), std::numeric_limits<Data_>::quiet_NaN());
        auto host = (my_eng() % MULT) * my_n;
        std::copy_n(ptr, my_n, host);
        return host;
    }
};
#endif

template<typename Index_, typename Data_>
class SimpleMatrixRandomAccessExtractor final : public RandomAccessExtractor<Index_, Data_> {
public:
    SimpleMatrixRandomAccessExtractor(const SimpleMatrix<Index_, Data_>& parent) : 
        my_parent(parent) 
#ifdef KMEANS_DEBUG_POINTER
        , my_buffer(my_parent.num_dimensions())
#endif
    {}

private:
    const SimpleMatrix<Index_, Data_>& my_parent;
#ifdef KMEANS_DEBUG_POINTER
    Buffer<Data_> my_buffer;
#endif

public:
    const Data_* get_observation(const Index_ i) {
        const auto output = my_parent.my_data + sanisizer::product_unsafe<std::size_t>(i, my_parent.my_num_dim);
#ifndef KMEANS_DEBUG_POINTER
        return output;
#else
        return my_buffer.update(output);
#endif
    }
};

template<typename Index_, typename Data_>
class SimpleMatrixConsecutiveAccessExtractor final : public ConsecutiveAccessExtractor<Index_, Data_> {
public:
    SimpleMatrixConsecutiveAccessExtractor(const SimpleMatrix<Index_, Data_>& parent, const Index_ start) :
        my_parent(parent),
        my_position(start) 
#ifdef KMEANS_DEBUG_POINTER
        , my_buffer(my_parent.num_dimensions())
#endif
    {}

private:
    const SimpleMatrix<Index_, Data_>& my_parent;
    Index_ my_position;
#ifdef KMEANS_DEBUG_POINTER
    Buffer<Data_> my_buffer;
#endif

public:
    const Data_* get_observation() {
        const auto output = my_parent.my_data + sanisizer::product_unsafe<std::size_t>(my_position++, my_parent.my_num_dim);
#ifndef KMEANS_DEBUG_POINTER
        return output;
#else
        return my_buffer.update(output);
#endif
    }
};

template<typename Index_, typename Data_>
class SimpleMatrixIndexedAccessExtractor final : public IndexedAccessExtractor<Index_, Data_> {
public:
    SimpleMatrixIndexedAccessExtractor(const SimpleMatrix<Index_, Data_>& parent, const Index_* const sequence) :
        my_parent(parent),
        my_sequence(sequence)
#ifdef KMEANS_DEBUG_POINTER
        , my_buffer(my_parent.num_dimensions())
#endif
    {}

private:
    const SimpleMatrix<Index_, Data_>& my_parent;
    const Index_* my_sequence;
    std::size_t my_position = 0;
#ifdef KMEANS_DEBUG_POINTER
    Buffer<Data_> my_buffer;
#endif

public:
    const Data_* get_observation() {
        const auto output = my_parent.my_data + sanisizer::product_unsafe<std::size_t>(my_sequence[my_position++], my_parent.my_num_dim);
#ifndef KMEANS_DEBUG_POINTER
        return output;
#else
        return my_buffer.update(output);
#endif
    }
};
/**
 * @endcond
 */

/**
 * @brief A simple matrix of observations.
 *
 * This defines a simple column-major matrix of observations where the columns are observations and the rows are dimensions.
 *
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Data_ Numeric type of the data.
 */
template<typename Index_, typename Data_>
class SimpleMatrix final : public Matrix<Index_, Data_> {
public:
    /**
     * @param num_dimensions Number of dimensions.
     * @param num_observations Number of observations.
     * @param[in] data Pointer to an array of length equal to the product of `num_dimensions` and `num_observations`, containing a column-major matrix of observation data.
     * It is expected that the array will not be deallocated during the lifetime of this `SimpleMatrix` instance.
     */
    SimpleMatrix(const std::size_t num_dimensions, const Index_ num_observations, const Data_* const data) :
        my_num_dim(num_dimensions), my_num_obs(num_observations), my_data(data) {}

private:
    std::size_t my_num_dim;
    Index_ my_num_obs;
    const Data_* my_data;
    friend class SimpleMatrixRandomAccessExtractor<Index_, Data_>;
    friend class SimpleMatrixConsecutiveAccessExtractor<Index_, Data_>;
    friend class SimpleMatrixIndexedAccessExtractor<Index_, Data_>;

public:
    Index_ num_observations() const {
        return my_num_obs;
    }

    std::size_t num_dimensions() const {
        return my_num_dim;
    }

public:
    std::unique_ptr<RandomAccessExtractor<Index_, Data_> > new_extractor() const {
        return new_known_extractor();
    }

    std::unique_ptr<ConsecutiveAccessExtractor<Index_, Data_> > new_extractor(const Index_ start, const Index_ length) const {
        return new_known_extractor(start, length);
    }

    std::unique_ptr<IndexedAccessExtractor<Index_, Data_> > new_extractor(const Index_* sequence, const std::size_t length) const {
        return new_known_extractor(sequence, length);
    }

public:
    /**
     * Override to assist devirtualization when creating a `RandomAccessExtractor`.
     */
    auto new_known_extractor() const {
        return std::make_unique<SimpleMatrixRandomAccessExtractor<Index_, Data_> >(*this);
    }

    /**
     * Override to assist devirtualization when creating a `ConsecutiveAccessExtractor`.
     */
    auto new_known_extractor(const Index_ start, const Index_) const {
        return std::make_unique<SimpleMatrixConsecutiveAccessExtractor<Index_, Data_> >(*this, start);
    }

    /**
     * Override to assist devirtualization when creating a `IndexedAccessExtractor`.
     */
    auto new_known_extractor(const Index_* sequence, const std::size_t) const {
        return std::make_unique<SimpleMatrixIndexedAccessExtractor<Index_, Data_> >(*this, sequence);
    }
};

}

#endif
