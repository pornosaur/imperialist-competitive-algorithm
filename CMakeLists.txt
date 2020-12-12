cmake_minimum_required(VERSION 3.15)
project(imperialist_competitive_algorithm)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

find_package( OpenCL REQUIRED)

SET(LIBS OpenCL::OpenCL tbb config++)

set(INCLUDE_DIR
        include/ica/imperialist_alg.h
        include/ica/solver_serial.h
        include/ica/solver_smp.h
        include/ica/solver_opencl.h

        include/ica/iface/ICASolver.h
        include/ica/iface/utils.h

        include/ica/benchmarks/ICABenchmarkFunctions.h)

set(SOURCE_DIR
        src/imperialist_alg.cpp
        src/solver_serial.cpp
        src/solver_smp.cpp
        src/solver_opencl.cpp

        src/iface/ICASolver.cpp

        src/benchmarks/ICABenchmarkFunctions.cpp)

add_executable(${PROJECT_NAME} main.cpp ${SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} ${LIBS})