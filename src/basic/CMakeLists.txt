add_executable(check_mat_flags check_mat_flags.cpp)
target_link_libraries(check_mat_flags ${OpenCV_LIBS})

add_executable(gaussian_blur gaussian_blur.cpp)
target_link_libraries(gaussian_blur ${OpenCV_LIBS})