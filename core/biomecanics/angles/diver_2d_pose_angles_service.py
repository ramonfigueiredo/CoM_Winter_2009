import os
import time

import numpy as np

from core.biomecanics.angles.diver_2d_pose_angles_utils import get_segment_angle_in_degrees, \
    get_joint_angle_in_degrees, save_output_video_with_the_angles_in_the_diver_2D_pose
from diving_analysis.models import OriginalPredictedDiver2DPose, EnhancedPredictedDiver2DPose


def diver_angles_estimation(show_video_frame, verbose, video):
    diver_2d_results_folder = 'diver_2d_pose_results'

    # **********************************
    # 1) Get the original diver's 2D poses
    # **********************************
    print("\n1) Get the original diver's 2D poses")
    start_time = time.time()

    original_diver_2d_pose_list = OriginalPredictedDiver2DPose.objects.filter(video=video)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 2) Compute and save angles in the original diver's 2D poses
    # **********************************
    print("\n2) Compute and save angles in the original diver's 2D poses")
    start_time = time.time()

    compute_and_save_angles_in_the_diver_2d_pose(original_diver_2d_pose_list)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 3) Save output video with the angles in the original diver 2d pose
    # **********************************
    print("\n3) Save output video with the angles in the original diver 2d pose")
    start_time = time.time()

    output_video_name_suffix = 'angles_in_the_original_diver_2D_pose'
    output_mp4_video_full_path = \
        save_output_video_with_the_angles_in_the_diver_2D_pose(
            original_diver_2d_pose_list, video, verbose, diver_2d_results_folder, output_video_name_suffix,
            show_video_frame, window_name="Angles in the original diver's 2D pose")

    # Associate video file with video object (original diver's 2D pose)
    video.video_with_angles_in_the_original_diver_2D_pose = \
        'videos' + os.sep + diver_2d_results_folder + os.sep + '{}' \
            .format(os.path.basename(output_mp4_video_full_path))
    video.save()

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 4) Get the enhanced diver's 2D poses
    # **********************************
    print("\n4) Get the enhanced diver's 2D poses")
    start_time = time.time()

    enhanced_diver_2d_pose_list = EnhancedPredictedDiver2DPose.objects.filter(video=video)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 2) Compute and save angles in the enhanced diver's 2D poses
    # **********************************
    print("\n2) Compute and save angles in the enhanced diver's 2D poses")
    start_time = time.time()

    compute_and_save_angles_in_the_diver_2d_pose(enhanced_diver_2d_pose_list)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 3) Save output video with the angles in the enhanced diver 2d pose
    # **********************************
    print("\n3) Save output video with the angles in the enhanced diver 2d pose")
    start_time = time.time()

    output_video_name_suffix = 'angles_in_the_enhanced_diver_2D_pose'
    output_mp4_video_full_path = \
        save_output_video_with_the_angles_in_the_diver_2D_pose(
            enhanced_diver_2d_pose_list, video, verbose, diver_2d_results_folder, output_video_name_suffix,
            show_video_frame, window_name="Angles in the enhanced diver's 2D pose")

    # Associate video file with video object (enhanced diver's 2D pose)
    video.video_with_angles_in_the_enhanced_diver_2D_pose = \
        'videos' + os.sep + diver_2d_results_folder + os.sep + '{}' \
            .format(os.path.basename(output_mp4_video_full_path))
    video.save()

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))


def compute_and_save_angles_in_the_diver_2d_pose(predicted_diver_2d_pose_list):

    for predicted_diver_2d_pose in predicted_diver_2d_pose_list:
        '''Get the angle of two points (segment angle).'''

        # Pred L Upper arm angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_LSho,
            pt1_y=predicted_diver_2d_pose.pred_y_LSho,
            pt2_x=predicted_diver_2d_pose.pred_x_LElb,
            pt2_y=predicted_diver_2d_pose.pred_y_LElb
        )
        predicted_diver_2d_pose.pred_L_Upper_arm_angle = angle_in_degree

        # Pred L Upper arm angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_RSho,
            pt1_y=predicted_diver_2d_pose.pred_y_RSho,
            pt2_x=predicted_diver_2d_pose.pred_x_RElb,
            pt2_y=predicted_diver_2d_pose.pred_y_RElb
        )
        predicted_diver_2d_pose.pred_R_Upper_arm_angle = angle_in_degree

        # Pred L Forearm angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_LElb,
            pt1_y=predicted_diver_2d_pose.pred_y_LElb,
            pt2_x=predicted_diver_2d_pose.pred_x_LWri,
            pt2_y=predicted_diver_2d_pose.pred_y_LWri
        )
        predicted_diver_2d_pose.pred_L_Forearm_angle = angle_in_degree

        # Pred R Forearm angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_RElb,
            pt1_y=predicted_diver_2d_pose.pred_y_RElb,
            pt2_x=predicted_diver_2d_pose.pred_x_RWri,
            pt2_y=predicted_diver_2d_pose.pred_y_RWri
        )
        predicted_diver_2d_pose.pred_R_Forearm_angle = angle_in_degree

        # Pred L Thigh angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_LHip,
            pt1_y=predicted_diver_2d_pose.pred_y_LHip,
            pt2_x=predicted_diver_2d_pose.pred_x_LKne,
            pt2_y=predicted_diver_2d_pose.pred_y_LKne
        )
        predicted_diver_2d_pose.pred_L_Thigh_angle = angle_in_degree

        # Pred R Thigh angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_RHip,
            pt1_y=predicted_diver_2d_pose.pred_y_RHip,
            pt2_x=predicted_diver_2d_pose.pred_x_RKne,
            pt2_y=predicted_diver_2d_pose.pred_y_RKne
        )
        predicted_diver_2d_pose.pred_R_Thigh_angle = angle_in_degree

        # Pred L Shank angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_LKne,
            pt1_y=predicted_diver_2d_pose.pred_y_LKne,
            pt2_x=predicted_diver_2d_pose.pred_x_LAnk,
            pt2_y=predicted_diver_2d_pose.pred_y_LAnk
        )
        predicted_diver_2d_pose.pred_L_Shank_angle = angle_in_degree

        # Pred R Shank angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_RKne,
            pt1_y=predicted_diver_2d_pose.pred_y_RKne,
            pt2_x=predicted_diver_2d_pose.pred_x_RAnk,
            pt2_y=predicted_diver_2d_pose.pred_y_RAnk
        )
        predicted_diver_2d_pose.pred_R_Shank_angle = angle_in_degree

        # Pred L Trunk angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_LSho,
            pt1_y=predicted_diver_2d_pose.pred_y_LSho,
            pt2_x=predicted_diver_2d_pose.pred_x_LHip,
            pt2_y=predicted_diver_2d_pose.pred_y_LHip
        )
        predicted_diver_2d_pose.pred_L_Trunk_angle = angle_in_degree

        # Pred R Trunk angle
        angle_in_degree = get_segment_angle_in_degrees(
            pt1_x=predicted_diver_2d_pose.pred_x_RSho,
            pt1_y=predicted_diver_2d_pose.pred_y_RSho,
            pt2_x=predicted_diver_2d_pose.pred_x_RHip,
            pt2_y=predicted_diver_2d_pose.pred_y_RHip
        )
        predicted_diver_2d_pose.pred_R_Trunk_angle = angle_in_degree

        '''Get the angle of three points (joint angle).'''

        # Pred L Elbow angle
        angle_in_degree = get_joint_angle_in_degrees(
            pt1=np.array([predicted_diver_2d_pose.pred_x_LSho, predicted_diver_2d_pose.pred_y_LSho]),
            pt2=np.array([predicted_diver_2d_pose.pred_x_LElb, predicted_diver_2d_pose.pred_y_LElb]),
            pt3=np.array([predicted_diver_2d_pose.pred_x_LWri, predicted_diver_2d_pose.pred_y_LWri])
        )
        predicted_diver_2d_pose.pred_L_Elbow_angle = angle_in_degree

        # Pred R Elbow angle
        angle_in_degree = get_joint_angle_in_degrees(
            pt1=np.array([predicted_diver_2d_pose.pred_x_RSho, predicted_diver_2d_pose.pred_y_RSho]),
            pt2=np.array([predicted_diver_2d_pose.pred_x_RElb, predicted_diver_2d_pose.pred_y_RElb]),
            pt3=np.array([predicted_diver_2d_pose.pred_x_RWri, predicted_diver_2d_pose.pred_y_RWri])
        )
        predicted_diver_2d_pose.pred_R_Elbow_angle = angle_in_degree

        # Pred L Shoulder angle
        angle_in_degree = get_joint_angle_in_degrees(
            pt1=np.array([predicted_diver_2d_pose.pred_x_LElb, predicted_diver_2d_pose.pred_y_LElb]),
            pt2=np.array([predicted_diver_2d_pose.pred_x_LSho, predicted_diver_2d_pose.pred_y_LSho]),
            pt3=np.array([predicted_diver_2d_pose.pred_x_LHip, predicted_diver_2d_pose.pred_y_LHip])
        )
        predicted_diver_2d_pose.pred_L_Shoulder_angle = angle_in_degree

        # Pred R Shoulder angle
        angle_in_degree = get_joint_angle_in_degrees(
            pt1=np.array([predicted_diver_2d_pose.pred_x_RElb, predicted_diver_2d_pose.pred_y_RElb]),
            pt2=np.array([predicted_diver_2d_pose.pred_x_RSho, predicted_diver_2d_pose.pred_y_RSho]),
            pt3=np.array([predicted_diver_2d_pose.pred_x_RHip, predicted_diver_2d_pose.pred_y_RHip])
        )
        predicted_diver_2d_pose.pred_R_Shoulder_angle = angle_in_degree

        # Pred L Hip angle
        angle_in_degree = get_joint_angle_in_degrees(
            pt1=np.array([predicted_diver_2d_pose.pred_x_LSho, predicted_diver_2d_pose.pred_y_LSho]),
            pt2=np.array([predicted_diver_2d_pose.pred_x_LHip, predicted_diver_2d_pose.pred_y_LHip]),
            pt3=np.array([predicted_diver_2d_pose.pred_x_LKne, predicted_diver_2d_pose.pred_y_LKne])
        )
        predicted_diver_2d_pose.pred_L_Hip_angle = angle_in_degree

        # Pred R Hip angle
        angle_in_degree = get_joint_angle_in_degrees(
            pt1=np.array([predicted_diver_2d_pose.pred_x_RSho, predicted_diver_2d_pose.pred_y_RSho]),
            pt2=np.array([predicted_diver_2d_pose.pred_x_RHip, predicted_diver_2d_pose.pred_y_RHip]),
            pt3=np.array([predicted_diver_2d_pose.pred_x_RKne, predicted_diver_2d_pose.pred_y_RKne])
        )
        predicted_diver_2d_pose.pred_R_Hip_angle = angle_in_degree

        # Pred L Knee angle
        angle_in_degree = get_joint_angle_in_degrees(
            pt1=np.array([predicted_diver_2d_pose.pred_x_LHip, predicted_diver_2d_pose.pred_y_LHip]),
            pt2=np.array([predicted_diver_2d_pose.pred_x_LKne, predicted_diver_2d_pose.pred_y_LKne]),
            pt3=np.array([predicted_diver_2d_pose.pred_x_LAnk, predicted_diver_2d_pose.pred_y_LAnk])
        )
        predicted_diver_2d_pose.pred_L_Knee_angle = angle_in_degree

        # Pred R Knee angle
        angle_in_degree = get_joint_angle_in_degrees(
            pt1=np.array([predicted_diver_2d_pose.pred_x_RHip, predicted_diver_2d_pose.pred_y_RHip]),
            pt2=np.array([predicted_diver_2d_pose.pred_x_RKne, predicted_diver_2d_pose.pred_y_RKne]),
            pt3=np.array([predicted_diver_2d_pose.pred_x_RAnk, predicted_diver_2d_pose.pred_y_RAnk])
        )
        predicted_diver_2d_pose.pred_R_Knee_angle = angle_in_degree

        # Update pose
        predicted_diver_2d_pose.save()
