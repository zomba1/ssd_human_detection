cmake_minimum_required(VERSION 3.16)

project(ssd_human_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ssd_human_detection main.cpp)

include(GNUInstallDirs)
install(TARGETS ssd_human_detection
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

target_link_libraries(ssd_human_detection PRIVATE ${OpenCV_LIBS})
