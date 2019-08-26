set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(decompose_homography decompose_homography.cpp)
target_link_libraries(decompose_homography
    ${OpenCV_LIBS}
)


add_executable(pose_from_homography pose_from_homography.cpp)
target_link_libraries(pose_from_homography
    ${OpenCV_LIBS}
    learnCV
)


add_executable(perspective_correction perspective_correction.cpp)
target_link_libraries(perspective_correction
    ${OpenCV_LIBS}
)

add_executable(homography_from_camera_displacement homography_from_camera_displacement.cpp)
target_link_libraries(homography_from_camera_displacement
    ${OpenCV_LIBS}
)
