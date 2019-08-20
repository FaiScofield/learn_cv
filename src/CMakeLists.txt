
add_subdirectory(calibration)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(learnCV SHARED
    calibration/Chessboard.cc
    )
target_link_libraries(learnCV
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    )


