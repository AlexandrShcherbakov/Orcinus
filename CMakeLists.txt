cmake_minimum_required(VERSION 3.7)
project(Orcinus)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -pedantic -fopenmp")


include_directories("~/InpharkhLib")
include_directories("include")

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    link_directories("~/InpharkhLib/Hors/cmake-build-debug")
else()
    link_directories("~/InpharkhLib/Hors/build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

link_directories("~/3rdPartyLibs/pugixml-1.8/build")

add_executable(Orcinus
        include/formfactors.h
        include/GeometryTypes.h
        include/tesselation.h
        src/main.cpp
        src/formfactors.cpp
        src/tesselation.cpp
        src/RadiosityComputation.cpp
        include/RadiosityComputation.h)

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
add_definitions(${GLUT_DEFINITIONS})

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
add_definitions(${OpenGL_DEFINITIONS})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
add_definitions(${GLEW_DEFINITIONS})

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(${GLM_INCLUDE_DIRS})
include_directories("~/3rdPartyLibs")

target_link_libraries(Orcinus
        "libHors.a"
        "libpugixml.a"
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARY}
        ${GLEW_LIBRARY}
        ${Boost_LIBRARIES}
)