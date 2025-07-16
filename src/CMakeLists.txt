# add_subdirectory(aruco)
add_subdirectory(basic)
# add_subdirectory(calibration)
# add_subdirectory(homography)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
if(UNIX)
    add_library(learnCV SHARED
        ${PROJECT_SOURCE_DIR}/include/utility.hpp
        calibration/Chessboard.cc
    #    calibration/ExtrinsicCalibrator.cc
        )
    target_link_libraries(learnCV
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
    #    ${CSPARSE_LIBRARY}
    #    ${G2O_LIBS}
        )
endif()