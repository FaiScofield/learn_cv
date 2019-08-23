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
