include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

option(CODE_COVERAGE "Enable coverage testing for tests" OFF)
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

macro(decorate_executable target)
    target_link_libraries(
        ${target}
        gtest_main
        kmeans
    )

    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)

    if(DO_CODE_COVERAGE)
        target_compile_options(${target} PRIVATE -O0 -g --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${target})
endmacro()

add_executable(
    libtest 
    src/compute_centroids.cpp
    src/compute_wcss.cpp
    src/SimpleMatrix.cpp
    src/InitializeNone.cpp
    src/InitializeRandom.cpp
    src/InitializeKmeanspp.cpp
    src/InitializeVariancePartition.cpp
    src/QuickSearch.cpp
    src/is_edge_case.cpp
    src/remove_unused_centers.cpp
    src/RefineLloyd.cpp
    src/RefineHartiganWong.cpp
    src/RefineMiniBatch.cpp
    src/kmeans.cpp
)
decorate_executable(libtest)
target_compile_definitions(libtest PRIVATE KMEANS_DEBUG_POINTER_USAGE=1) 

# Get some coverage on the regular behavior of a non-debug-mode SimpleMatrix.
add_executable(
    mattest 
    src/SimpleMatrix.cpp
)
decorate_executable(mattest)

add_executable(
    cuspartest
    src/InitializeKmeanspp.cpp
    src/RefineLloyd.cpp
    src/RefineHartiganWong.cpp
    src/RefineMiniBatch.cpp
)
decorate_executable(cuspartest)
target_compile_definitions(cuspartest PRIVATE TEST_CUSTOM_PARALLEL=1)
