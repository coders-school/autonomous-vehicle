# OpenCV include directories
set(OPENCV_INCLUDE_DIRS
    ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
    ${OPENCV_MODULE_opencv_core_LOCATION}/include
    ${OPENCV_MODULE_opencv_calib3d_LOCATION}/include/
    ${OPENCV_MODULE_opencv_features2d_LOCATION}/include/
    ${OPENCV_MODULE_opencv_highgui_LOCATION}/include/
    ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include/
    ${OPENCV_MODULE_opencv_videoio_LOCATION}/include/
)

# OpenCV libraries
set(OPENCV_LIBRARIES
    opencv_core 
    opencv_calib3d
    opencv_features2d
    opencv_highgui
    opencv_imgcodecs
    opencv_videoio
)
