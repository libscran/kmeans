# C++ library for k-means

![Unit tests](https://github.com/libscran/kmeans/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/libscran/kmeans/actions/workflows/doxygenate.yaml/badge.svg)
![stats comparison](https://github.com/libscran/kmeans/actions/workflows/compare-kmeans.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/libscran/kmeans/branch/master/graph/badge.svg?token=7S231XHC0Q)](https://codecov.io/gh/libscran/kmeans)

## Overview

This repository contains a header-only C++ library for k-means clustering.
Initialization can be performed with user-supplied centers, random selection of points, weighted sampling with kmeans++ (Arthur and Vassilvitskii, 2007) or variance partitioning (Su and Dy, 2007).
Refinement can be performed using the Hartigan-Wong method, Lloyd's algorithm or a custom mini-batch implementation.
The Hartigan-Wong implementation is derived from the Fortran code in the R **stats** package, heavily refactored for more idiomatic C++.

## Quick start

**kmeans** is a header-only library, so it can be easily used by just `#include`ing the relevant source files and running `compute()`:

```cpp
#include "kmeans/Kmeans.hpp"

int ndim = 5;
int nobs = 1000;
std::vector<double> matrix(ndim * nobs); // column-major ndim x nobs matrix of coordinates

// Wrap your matrix in a SimpleMatrix.
kmeans::SimpleMatrix<
    int, /* type of the column index */
    double /* type of the data */
> kmat(ndim, nobs, matrix.data());

auto res = kmeans::compute(
    kmat,
    // initialize with kmeans++
    kmeans::InitializeKmeanspp<
        /* column index type */ int,
        /* input matrix data type */ double, 
        /* cluster ID type */ int, 
        /* centroid type */ double
    >(),
    // refine with Lloyd's algorithm
    kmeans::RefineLloyd<
        /* column index type */ int,
        /* input matrix data type */ double, 
        /* cluster ID type */ int, 
        /* centroid type */ double
    >(),
    ncenters 
);

res.centers; // Matrix of centroid coordinates, stored in column-major format
res.clusters; // Vector of cluster assignments
res.details; // Details from the clustering algorithm
```

See the [reference documentation](https://ltla.github.io/CppKmeans) for more details.

## Changing parameters 

We can tune the clustering by passing options into the constructors of the relevant classes:

```cpp
kmeans::InitializeVariancePartitionOptions vp_opt;
vp_opt.optimize_partition = false;
kmeans::InitializeVariancePartition<int, double, int, double> vp(vp_opt);

kmeans::RefineLloydOptions ll_opt;
ll_opt.max_iterations = 10;
ll_opt.num_threads = 3;
kmeans::RefineLloyd<int, double, int, double> ll(ll_opt);

auto res2 = kmeans::compute(kmat, pp, ll, ncenters);
```

The initialization and refinement classes can themselves be swapped at run-time via pointers to their respective interfaces.
This design also allows the **kmeans** library to be easily extended to additional methods from third-party developers.

```cpp
std::unique_ptr<kmeans::Initialize<int, double, int, double> > init_ptr;
if (init_method == "random") {
    init_ptr.reset(new kmeans::InitializeRandom<int, double, int, double>);
} else if (init_method == "kmeans++") {
    kmeans::InitializeKmeansppOptions opt;
    opt.seed = 42;
    init_ptr.reset(new kmeans::InitializeKmeanspp<int, double, int, double>(opt));
} else {
    // do something else
}

std::unique_ptr<kmeans::Refine<int, double, int, double> > ref_ptr;
if (ref_method == "random") {
    kmeans::RefineLloydOptions opt;
    opt.max_iterations = 10;
    ref_ptr.reset(new kmeans::RefineLloyd<int, double, int, double>(opt));
} else {
    kmeans::RefineHartiganWongOptions opt;
    opt.max_iterations = 100;
    opt.max_quick_transfer_iterations = 1000;
    ref_ptr.reset(new kmeans::RefineHartiganWong<int, double, int, double>(opt));
}

auto res3 = kmeans::compute(kmat, *init_ptr, *ref_ptr, ncenters);
```

Template parameters can also be altered to control the input and output data types.
As shown above, these should be set consistently for all classes used in `compute()`. 
While `int` and `double` are suitable for most cases, advanced users may wish to use other types.
For example, we might consider the following parametrization for various reasons:

```cpp
kmeans::InitializeKmeanspp<
    /* If our input data has too many observations to fit into an 'int', we
     * might need to use a 'size_t' instead.
     */
    size_t,

    /* Perhaps our input data is in single-precision floating point to save
     * space and to speed up processing.
     */
    float, 

    /* If we know that we will never ask for more than 255 clusters, we can use
     * a smaller integer for the cluster IDs to save space.
     */
    uint8_t, 

    /* We still want our centroids and distances to be computed in high
     * precision, even though the input data is only single precision.
     */
    double 
> initpp();
```

## Other bits and pieces

If we want the within-cluster sum of squares, this can be easily calculated from the output of `compute()`:

```cpp
std::vector<double> wcss(ncenters);
kmeans::compute_wcss(
    kmat, 
    ncenters, 
    res.centers.data(), 
    res.clusters.data(), 
    wcss.data()
);
```

If we already allocated arrays for the centroids and clusters, we can fill the arrays directly.
This allows us to skip a copy when interfacing with other languages that manage their own memory (e.g., R, Python).

```cpp
std::vector<double> centers(ndim * ncenters);
std::vector<int> clusters(nobs);

auto deets = kmeans::compute(
    kmat,
    kmeans::InitializeRandom(), // random initialization
    kmeans::RefineHartiganWong(), // refine with Hartigan-Wong 
    ncenters 
    centers.data(),
    clusters.data()
);
```

## Building projects 

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/libscran/CppKmeans
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(kmeans)
```

Then you can link to **kmeans** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe ltla::kmeans)

# For libaries
target_link_libraries(mylib INTERFACE ltla::kmeans)
```

By default, this will use `FetchContent` to fetch all external dependencies. 
Applications are advised to pin the versions of each dependency for stability - see [`extern/CMakeLists.txt`](extern/CMakeLists.txt) for suggested versions.
If you want to install them manually, use `-DKMEANS_FETCH_EXTERN=OFF`.

### CMake with `find_package()`

To install the library, clone an appropriate version of this repository and run:

```sh
mkdir build && cd build
cmake .. -DKMEANS_TESTS=OFF
cmake --build . --target install
```

Then we can use `find_package()` as usual:

```cmake
find_package(ltla_kmeans CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE ltla::kmeans)
```

Again, this will automatically acquire all its dependencies, see recommendations above.

### Manual

If you're not using CMake, the simple approach is to just copy the files in `include/` - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt). 

## References

Hartigan, J. A. and Wong, M. A. (1979).
Algorithm AS 136: A K-means clustering algorithm.
_Applied Statistics_ 28, 100-108.

Arthur, D. and Vassilvitskii, S. (2007). 
k-means++: the advantages of careful seeding.
_Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms_, 1027-1035.

Su, T. and Dy, J. G. (2007).
In Search of Deterministic Methods for Initializing K-Means and Gaussian Mixture Clustering,
_Intelligent Data Analysis_ 11, 319-338.
