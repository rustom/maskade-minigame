if(APPLE)
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)
endif()

cmake_minimum_required(VERSION 3.1.3)
project(maskade-classifier VERSION 0.1.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent.git
  GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
FetchContent_MakeAvailable(json)
if (NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Link the correct path to your TensorFlow C library
find_library(TENSORFLOW_LIB tensorflow HINT /Users/rustomichhaporia/GitHub/libtensorflow-cpu-darwin-x86_64-2.3.1/lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)

# Link the correct path to your OpenCV library
set(OpenCV_DIR "/Users/rustomichhaporia/GitHub/build_opencv")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

add_library(opencv INTERFACE)
include_directories(${OpenCV_INCLUDE_DIRS})

get_filename_component(CINDER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)
get_filename_component(APP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/" ABSOLUTE)

# Link the correct path to your Cinder library
include("${CINDER_PATH}/proj/cmake/modules/cinderMakeApp.cmake")
include("/Users/rustomichhaporia/GitHub/Cinder/lib/macosx/DEBUG/cinderConfig.cmake")

list(APPEND SOURCE_FILES   
    src/maskade_classifier.cpp
)

ci_make_app(
        APP_NAME        maskade-visualizer
        CINDER_PATH     ${CINDER_PATH}
        SOURCES         apps/maskade_app.cpp ${SOURCE_FILES}
        INCLUDES        include  
        LIBRARIES       nlohmann_json::nlohmann_json
)   

target_link_libraries(opencv INTERFACE ${OpenCV_LIBS})
target_link_libraries(maskade-visualizer ${OpenCV_LIBS})

# Link the correct path to your cppflow library
target_include_directories(maskade-visualizer PRIVATE /Users/rustomichhaporia/GitHub/cppflow2/include /Users/rustomichhaporia/GitHub/libtensorflow-cpu-darwin-x86_64-2.3.1/include)
target_link_libraries (maskade-visualizer "${TENSORFLOW_LIB}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)