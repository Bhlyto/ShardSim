cmake_minimum_required(VERSION 3.24)
project(shardsim LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SHARDSIM_DETERMINISTIC_MODE "Enable deterministic behavior where possible" ON)

find_package(MPI QUIET)

add_library(shardsim_core
    src/decision_core/policy.cpp
    src/io/config_loader.cpp
    src/mesh/mesh.cpp
    src/mesh/mesh3d.cpp
    src/metrics/metrics.cpp
    src/mpi/mpi_runtime.cpp
    src/orchestrator/orchestrator.cpp
    src/solver/heat_solver.cpp
    src/solver/heat_solver3d.cpp
)

target_include_directories(shardsim_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MPI_FOUND)
    target_compile_definitions(shardsim_core PUBLIC SHARDSIM_HAS_MPI=1)
    target_link_libraries(shardsim_core PUBLIC MPI::MPI_CXX)
else()
    target_compile_definitions(shardsim_core PUBLIC SHARDSIM_HAS_MPI=0)
endif()

if(SHARDSIM_DETERMINISTIC_MODE)
    target_compile_definitions(shardsim_core PUBLIC SHARDSIM_DETERMINISTIC_MODE=1)
else()
    target_compile_definitions(shardsim_core PUBLIC SHARDSIM_DETERMINISTIC_MODE=0)
endif()

add_executable(shardsim_cli src/main.cpp)
target_link_libraries(shardsim_cli PRIVATE shardsim_core)

install(TARGETS shardsim_cli RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_executable(shardsim_partition_halo_test tests/partition_halo_test.cpp)
    target_link_libraries(shardsim_partition_halo_test PRIVATE shardsim_core)

    add_executable(shardsim_kpi_guardrail_test tests/kpi_guardrail_test.cpp)
    target_link_libraries(shardsim_kpi_guardrail_test PRIVATE shardsim_core)

    add_executable(shardsim_decision_core_policy_test tests/decision_core_policy_test.cpp)
    target_link_libraries(shardsim_decision_core_policy_test PRIVATE shardsim_core)

    add_executable(shardsim_mesh3d_smoke_test tests/mesh3d_smoke_test.cpp)
    target_link_libraries(shardsim_mesh3d_smoke_test PRIVATE shardsim_core)

    add_executable(shardsim_policy_fail_test tests/policy_fail_test.cpp)
    target_link_libraries(shardsim_policy_fail_test PRIVATE shardsim_core)

    add_test(NAME shardsim_partition_halo_single COMMAND shardsim_partition_halo_test)
    add_test(NAME shardsim_kpi_guardrail_single COMMAND shardsim_kpi_guardrail_test)
    add_test(NAME shardsim_decision_core_policy_single COMMAND shardsim_decision_core_policy_test)
    add_test(NAME shardsim_mesh3d_smoke_single COMMAND shardsim_mesh3d_smoke_test)
    add_test(NAME shardsim_policy_wallclock_fail_single COMMAND shardsim_policy_fail_test wallclock)
    add_test(NAME shardsim_policy_memory_fail_single COMMAND shardsim_policy_fail_test memory)
    set_tests_properties(
        shardsim_policy_wallclock_fail_single
        shardsim_policy_memory_fail_single
        PROPERTIES
            WILL_FAIL TRUE)

    if(MPI_FOUND)
        add_test(
            NAME shardsim_partition_halo_mpi2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:shardsim_partition_halo_test>)
        add_test(
            NAME shardsim_kpi_guardrail_mpi2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:shardsim_kpi_guardrail_test>)
        add_test(
            NAME shardsim_policy_wallclock_fail_mpi2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:shardsim_policy_fail_test> wallclock)
        add_test(
            NAME shardsim_policy_memory_fail_mpi2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:shardsim_policy_fail_test> memory)
        set_tests_properties(
            shardsim_policy_wallclock_fail_mpi2
            shardsim_policy_memory_fail_mpi2
            PROPERTIES
                WILL_FAIL TRUE)
    endif()
endif()
