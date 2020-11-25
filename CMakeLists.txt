# set(CMAKE_C_COMPILER /usr/bin/gcc)
# set(CMAKE_CXX_COMPILER /usr/bin/g++)
if(APPLE)
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)
endif()

cmake_minimum_required(VERSION 3.1.3)
project(ideal-gas VERSION 0.1.0)

# if(APPLE)
# set(CMAKE_FIND_FRAMEWORK LAST)
# set(CMAKE_FIND_APPBUNDLE LAST)
# endif()

# Let's nicely support folders in IDE's
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# include(CTest)
# enable_testing()

# # Warning flags
# if(MSVC)
#     # warning level 3 and all warnings as errors
#     add_compile_options(/W3 /WX)
# else()
#     # lots of warnings and all warnings as errors
#     add_compile_options(-Wall -Wpedantic -Werror)
# endif()

# include(FetchContent)

# FetchContent_Declare(
#   Catch2
#   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#   GIT_TAG        v2.13.1)

# FetchContent_MakeAvailable(Catch2)

# FetchContent_Declare(json
#   GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent.git
#   GIT_TAG v3.7.3)

# FetchContent_GetProperties(json)
# FetchContent_MakeAvailable(json)
# if(NOT json_POPULATED)
#   FetchContent_Populate(json)
#   add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
# endif()

# find_package( Boost 1.54 REQUIRED COMPONENTS system filesystem )
# list( APPEND CINDER_LIBS_DEPENDS ${Boost_LIBRARIES} )
# list( APPEND CINDER_INCLUDE_SYSTEM_PRIVATE ${Boost_INCLUDE_DIRS} )

# include(CheckCXXCompilerFlag) 

# Check for standard to use
# check_cxx_compiler_flag(-std=c++17 HAVE_FLAG_STD_CXX17)
# if(HAVE_FLAG_STD_CXX17)
#     # Have -std=c++17, use it
# else()
#     check_cxx_compiler_flag(-std=c++1z HAVE_FLAG_STD_CXX1Z)
#     if(HAVE_FLAG_STD_CXX1Z)
#         # Have -std=c++1z, use it
#     else()
#         # And so on and on...
#     endif()
# endif()

# LIBRARIES ( opencv_core opencv_highgui opencv_imgproc opencv_videoio )

find_library(TENSORFLOW_LIB tensorflow HINT /Users/rustomichhaporia/GitHub/libtensorflow-cpu-darwin-x86_64-2.3.1/lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)




set(OpenCV_DIR "/Users/rustomichhaporia/GitHub/build_opencv")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

add_library(opencv INTERFACE)
include_directories(${OpenCV_INCLUDE_DIRS})



get_filename_component(CINDER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)
get_filename_component(APP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/" ABSOLUTE)

include("${CINDER_PATH}/proj/cmake/modules/cinderMakeApp.cmake")
include("/Users/rustomichhaporia/GitHub/Cinder/lib/macosx/DEBUG/cinderConfig.cmake")
list(APPEND CORE_SOURCE_FILES   
    src/maskade_classifier.cpp
    # src/core/particle_container.cc
    # src/core/particle.cc
    # src/core/histogram.cc
)

list(APPEND VISUALIZER_SOURCE_FILES  
    # src/visualizer/ideal_gas_visualizer.cc
)

list(APPEND TEST_FILES      
    # test/test_main.cc
    # test/test_particle_container.cc
    # test/test_histogram.cc
)


ci_make_app(
        APP_NAME        maskade-visualizer
        CINDER_PATH     ${CINDER_PATH}
        SOURCES         apps/maskade_app.cpp ${CORE_SOURCE_FILES} ${VISUALIZER_SOURCE_FILES}
        INCLUDES        include  # blocks/Cinder-OpenCV-master/include
        # BLOCKS          Cinder-OpenCV-master 
)   

target_link_libraries(opencv INTERFACE ${OpenCV_LIBS})
target_link_libraries(maskade-visualizer ${OpenCV_LIBS})

target_include_directories(maskade-visualizer PRIVATE /Users/rustomichhaporia/GitHub/cppflow2/include/ /Users/rustomichhaporia/GitHub/libtensorflow-cpu-darwin-x86_64-2.3.1/include)
target_link_libraries (maskade-visualizer "${TENSORFLOW_LIB}")

# ci_make_app(
#         APP_NAME        maskade-test
#         CINDER_PATH     ${CINDER_PATH}
#         SOURCES         test/test_main.cc ${CORE_SOURCE_FILES} ${TEST_FILES}
#         INCLUDES        include
#         LIBRARIES       Catch2 
# )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)