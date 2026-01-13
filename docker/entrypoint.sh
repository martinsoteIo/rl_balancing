#!/bin/bash
set -e

# Source ROS 2 installation
source /opt/ros/${ROS_DISTRO}/setup.bash

exec "$@"
