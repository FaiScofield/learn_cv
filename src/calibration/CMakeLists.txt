
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

#add_executable(calibrateWithAruco calibrateWithAruco.cpp)
#target_link_libraries(calibrateWithAruco
#    ${OpenCV_LIBS}
#    ${Boost_LIBRARIES}
#)

add_executable(calibrateWithChessboard calibrateWithChessboard.cpp)
target_link_libraries(calibrateWithChessboard
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    learnCV
)

#add_executable(undistortImage undistortImage.cpp)
#target_link_libraries(undistortImage
#    ${OpenCV_LIBS}
#    ${Boost_LIBRARIES}
#)

#add_executable(calibrateWithChessboardMatlab calibrateWithChessboardMatlab.cpp)
#target_link_libraries(calibrateWithChessboardMatlab
#    ${OpenCV_LIBS}
#    ${Boost_LIBRARIES}
#)

#add_executable(extrinsicCalibrateBetweenCameraOdom extrinsicCalibrateBetweenCameraOdom.cpp)
#target_link_libraries(extrinsicCalibrateBetweenCameraOdom
#    learnCV
#    ${OpenCV_LIBS}
#    ${Boost_LIBRARIES}
#    )
