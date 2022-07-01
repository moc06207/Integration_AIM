cmake_minimum_required(VERSION 3.0.2)
project(INS_Integration)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  message_generation
)

add_message_files(
  FILES
  Ins.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  LIBRARIES INS_Integration
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(PROGRAMS
src/sync_3.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
