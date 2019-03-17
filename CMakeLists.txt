cmake_minimum_required(VERSION 3.7)
project(Orcinus)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -fopenmp")

set(CMAKE_PREFIX_PATH "/usr/lib64/cmake")

include_directories("../../../Projects/InpharkhLibs")
include_directories("include")
include_directories("../../../Projects/3rdPartyLibraries/glm")
include_directories("../../../Projects/3rdPartyLibraries/freeglut-3.0.0/include")
include_directories("../../../Projects/3rdPartyLibraries")
include_directories("../../../Projects/3rdPartyLibraries/glew-1.13.0/include")

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    link_directories("../../../Projects/InpharkhLibs/Hors/cmake-build-debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
else()
    link_directories("../../../Projects/InpharkhLibs/Hors/build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

link_directories("../../../Projects/3rdPartyLibraries/pugixml/build")
#link_directories("../3rdPartyLibraries/glew-1.13.0/build/lib")
link_directories("../../../Projects/3rdPartyLibraries/freeglut-3.0.0/Release/lib")
link_directories("../../../Projects/3rdPartyLibraries/glew-1.13.0/lib")

message("${LINK_DIRECTORIES}")

add_executable(Orcinus
        include/formfactors.h
        include/GeometryTypes.h
        include/tesselation.h
        src/main.cpp
        src/formfactors.cpp
        src/tesselation.cpp
        src/RadiosityComputation.cpp
        include/RadiosityComputation.h)

set(OpenGL_GL_PREFERENCE GLVND)
#set(GLUT_ROOT_PATH "../3rdPartyLibraries/freeglut-3.0.0")
#set(GLEW_INCLUDE_DIR "../3rdPartyLibraries/glew-1.13.0/include")
#set(GLEW_LIBRARY "../3rdPartyLibraries/glew-1.13.0/lib/")
set(GLEW_LIBRARY "libglew32.a")
set(BOOST_ROOT "../../../Projects/3rdPartyLibraries/boost_1_69_0")
set(BOOST_LIBRARYDIR "../../../Projects3rdPartyLibraries/boost_1_69_0/stage/lib")
set(CMAKE_PREFIX_PATH "../../../Projects/3rdPartyLibraries/embree")

#find_package(GLUT REQUIRED)
#include_directories(${GLUT_INCLUDE_DIRS})
#add_definitions(${GLUT_DEFINITIONS})

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
add_definitions(${OpenGL_DEFINITIONS})

#find_package(GLEW REQUIRED)
#include_directories(${GLEW_INCLUDE_DIRS})
#add_definitions(${GLEW_DEFINITIONS})

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(embree 3.0 REQUIRED)
include_directories(${EMBREE_INCLUDE_DIRS})

include_directories(${GLM_INCLUDE_DIRS})
include_directories("../3rdPartyLibraries")

target_link_libraries(Orcinus
        "libHors.a"
        "libpugixml.a"
        ${OPENGL_LIBRARIES}
        "-lfreeglut"
        "-lglew32"
#        "libglu32.a"
#        ${GLUT_LIBRARY}
#        ${GLEW_LIBRARY}
        ${Boost_LIBRARIES}
        ${EMBREE_LIBRARY}
)