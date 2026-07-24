#!/usr/bin/env python3
# Copyright (c) 2026, Unitree RL Lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Official Unitree G1 29DOF joint effort limits used for torque plots.

Source: Unitree Robotics unitree_ros
robots/g1_description/g1_29dof_rev_1_0.urdf
https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.urdf
"""

G1_29DOF_EFFORT_LIMIT_SOURCE = (
    "Unitree official unitree_ros robots/g1_description/g1_29dof_rev_1_0.urdf"
)
G1_29DOF_EFFORT_LIMIT_SOURCE_URL = (
    "https://github.com/unitreerobotics/unitree_ros/blob/master/"
    "robots/g1_description/g1_29dof_rev_1_0.urdf"
)

G1_29DOF_EFFORT_LIMITS_NM = {
    "left_hip_pitch_joint": 88.0,
    "right_hip_pitch_joint": 88.0,
    "waist_yaw_joint": 88.0,
    "left_hip_roll_joint": 139.0,
    "right_hip_roll_joint": 139.0,
    "left_hip_yaw_joint": 88.0,
    "right_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "right_knee_joint": 139.0,
    "left_ankle_pitch_joint": 35.0,
    "right_ankle_pitch_joint": 35.0,
    "left_ankle_roll_joint": 35.0,
    "right_ankle_roll_joint": 35.0,
    "waist_roll_joint": 35.0,
    "waist_pitch_joint": 35.0,
    "left_shoulder_pitch_joint": 25.0,
    "right_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "right_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "left_wrist_pitch_joint": 5.0,
    "right_wrist_pitch_joint": 5.0,
    "left_wrist_yaw_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}


def g1_29dof_effort_limit_nm(joint_name: str) -> float | None:
    return G1_29DOF_EFFORT_LIMITS_NM.get(joint_name)
