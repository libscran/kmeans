cmake_minimum_required(VERSION 3.24)

project(kmeans
    VERSION 4.0.4
    DESCRIPTION "K-means clustering in C++"
    LANGUAGES CXX)

add_library(kmeans INTERFACE)
add_library(ltla::kmeans ALIAS kmeans)

include(GNUInstallDirs)
target_include_directories(kmeans INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ltla_kmeans>)

target_compile_features(kmeans INTERFACE cxx_std_17)

# Dependencies
option(KMEANS_FETCH_EXTERN "Automatically fetch kmeans's external dependencies." ON)
if(KMEANS_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(ltla_aarand 1.1.0 CONFIG REQUIRED)
    find_package(ltla_subpar 0.4.0 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.1.3 CONFIG REQUIRED)
endif()

target_link_libraries(kmeans INTERFACE ltla::aarand ltla::subpar ltla::sanisizer)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(KMEANS_TESTS "Build kmeans's test suite." ON)
else()
    option(KMEANS_TESTS "Build kmeans's test suite." OFF)
endif()
if(KMEANS_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ltla_kmeans)

install(TARGETS kmeans
    EXPORT kmeansTargets)

install(EXPORT kmeansTargets
    FILE ltla_kmeansTargets.cmake
    NAMESPACE ltla::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_kmeans)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_kmeansConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_kmeans)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_kmeansConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ltla_kmeansConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_kmeansConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_kmeans)
