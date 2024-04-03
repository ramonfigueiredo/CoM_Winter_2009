import os
import time

from core.biomecanics.center_of_mass.center_of_mass_utils import get_keypoints_list, get_segment_CoM_location, \
    get_body_CoM_location
from core.biomecanics.center_of_mass.center_of_mass_utils import \
    save_output_video_with_the_diver_2d_pose_CoM_tracking_line
from core.biomecanics.motion.acceleration_utils import get_centred_acceleration_list, get_horizontal_acceleration_list, \
    get_vertical_acceleration_list
from core.biomecanics.motion.velocity_utils import get_centred_velocity_list, get_horizontal_velocity_list, \
    get_vertical_velocity_list
from core.signal_processing.signal_processing_utils import get_enhanced_data_list
from diving_analysis.models import EnhancedPredictedDiver2DPose, OriginalPredictedDiver2DPose


def diver_CoM_estimation(video):
    # **********************************
    # 1.1) Center of mass of the original diver 2D poses
    # **********************************

    print("\n1.1) Center of mass of the original diver 2D poses")

    # 1.1.1) Get the original diver 2D poses associated with the video
    print('1.1.1) Get the original diver 2D poses associated with the video')
    start_time = time.time()

    original_predicted_diver_2d_pose_list = OriginalPredictedDiver2DPose.objects.filter(video=video)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # 1.1.2) Save the center of mass for all the diver 2D poses
    print('1.1.2) Save the center of mass for all the diver 2D poses')
    start_time = time.time()

    save_diver_2d_poses_CoM(video, original_predicted_diver_2d_pose_list, is_enhanced_diver_2d_poses=False)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 1.2) Center of mass of the enhanced diver 2D poses
    # **********************************

    print("\n1.2) Center of mass of the enhanced diver 2D poses")

    # 1.2.1) Get the enhanced diver 2D poses associated with the video
    print("\n1.2.1) Get the enhanced diver 2D poses associated with the video")
    start_time = time.time()

    enhanced_predicted_diver_2d_pose_list = EnhancedPredictedDiver2DPose.objects.filter(video=video)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # 1.2.2) Save the center of mass for all the diver 2D poses
    print("\n1.2.2) Save the center of mass for all the diver 2D poses")
    start_time = time.time()

    save_diver_2d_poses_CoM(video, enhanced_predicted_diver_2d_pose_list, is_enhanced_diver_2d_poses=True)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))


def save_diver_2d_poses_CoM(video, diver_2d_pose_saved_list, is_enhanced_diver_2d_poses):

    # Get the center of mass of the original diver 2D poses
    filtered_x_body_CoM_np, filtered_y_body_CoM_np, \
        Head_X_CoM_np, Head_Y_CoM_np, \
        L_Shank_X_CoM_np, L_Shank_Y_CoM_np, R_Shank_X_CoM_np, R_Shank_Y_CoM_np, \
        L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np, R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np, \
        L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np, R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np, \
        L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np, R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np, \
        Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np = get_CoM_list(video, diver_2d_pose_saved_list)

    # Diver's 2D pose list to update
    diver_2d_pose_bulk_update_list = []

    for diver_2d_pose, diver_x_CoM, diver_y_CoM, \
            Head_CoM, Head_CoM, \
            L_Shank_x_CoM, L_Shank_y_CoM, R_Shank_x_CoM, R_Shank_y_CoM, \
            L_Thigh_x_CoM, L_Thigh_y_CoM, R_Thigh_x_CoM, R_Thigh_y_CoM, \
            L_Up_Arm_x_CoM, L_Up_Arm_y_CoM, R_Up_Arm_x_CoM, R_Up_Arm_y_CoM, \
            L_Forearm_x_CoM, L_Forearm_y_CoM, R_Forearm_x_CoM, R_Forearm_y_CoM, \
            Mid_Trunk_x_CoM, Mid_Trunk_y_CoM in \
            zip(diver_2d_pose_saved_list, filtered_x_body_CoM_np, filtered_y_body_CoM_np,
                Head_X_CoM_np, Head_Y_CoM_np,
                L_Shank_X_CoM_np, L_Shank_Y_CoM_np, R_Shank_X_CoM_np, R_Shank_Y_CoM_np,
                L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np, R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np,
                L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np, R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np,
                L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np, R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np,
                Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np):

        # Diver's Center of Mass (CoM)
        diver_2d_pose.diver_x_CoM = round(diver_x_CoM, 2)
        diver_2d_pose.diver_y_CoM = round(diver_y_CoM, 2)

        # Diver's Segment Center of Mass (CoM)
        # Segments: Head CoM, L/R Shank CoM, L/R Thigh CoM, L/R Up Arm CoM, L/R Forearm CoM, Mid Trunk CoM
        diver_2d_pose.Head_CoM = round(Head_CoM, 2)
        diver_2d_pose.Head_CoM = round(Head_CoM, 2)

        diver_2d_pose.L_Shank_x_CoM = round(L_Shank_x_CoM, 2)
        diver_2d_pose.L_Shank_y_CoM = round(L_Shank_y_CoM, 2)
        diver_2d_pose.R_Shank_x_CoM = round(R_Shank_x_CoM, 2)
        diver_2d_pose.R_Shank_y_CoM = round(R_Shank_y_CoM, 2)

        diver_2d_pose.L_Thigh_x_CoM = round(L_Thigh_x_CoM, 2)
        diver_2d_pose.L_Thigh_y_CoM = round(L_Thigh_y_CoM, 2)
        diver_2d_pose.R_Thigh_x_CoM = round(R_Thigh_x_CoM, 2)
        diver_2d_pose.R_Thigh_y_CoM = round(R_Thigh_y_CoM, 2)

        diver_2d_pose.L_Up_Arm_x_CoM = round(L_Up_Arm_x_CoM, 2)
        diver_2d_pose.L_Up_Arm_y_CoM = round(L_Up_Arm_y_CoM, 2)
        diver_2d_pose.R_Up_Arm_x_CoM = round(R_Up_Arm_x_CoM, 2)
        diver_2d_pose.R_Up_Arm_y_CoM = round(R_Up_Arm_y_CoM, 2)

        diver_2d_pose.L_Forearm_x_CoM = round(L_Forearm_x_CoM, 2)
        diver_2d_pose.L_Forearm_y_CoM = round(L_Forearm_y_CoM, 2)
        diver_2d_pose.R_Forearm_x_CoM = round(R_Forearm_x_CoM, 2)
        diver_2d_pose.R_Forearm_y_CoM = round(R_Forearm_y_CoM, 2)

        diver_2d_pose.Mid_Trunk_x_CoM = round(Mid_Trunk_x_CoM, 2)
        diver_2d_pose.Mid_Trunk_y_CoM = round(Mid_Trunk_y_CoM, 2)

        # Update the diver's 2D pose list
        diver_2d_pose_bulk_update_list.append(diver_2d_pose)

    # Store the updated diver 2D poses with CoM
    store_diver_2d_poses_with_CoM_on_the_database(diver_2d_pose_bulk_update_list, is_enhanced_diver_2d_poses)


def store_diver_2d_poses_with_CoM_on_the_database(diver_2d_pose_bulk_update_list, is_enhanced_diver_2d_poses):

    if diver_2d_pose_bulk_update_list:
        '''
        This method updates the provided list of objects into the database in an efficient manner
        '''
        if is_enhanced_diver_2d_poses:
            EnhancedPredictedDiver2DPose.objects.bulk_update(
                diver_2d_pose_bulk_update_list,
                ['diver_x_CoM', 'diver_y_CoM',
                 'Head_CoM', 'Head_CoM',
                 'L_Shank_x_CoM', 'L_Shank_y_CoM', 'R_Shank_x_CoM', 'R_Shank_y_CoM',
                 'L_Thigh_x_CoM', 'L_Thigh_y_CoM', 'R_Thigh_x_CoM', 'R_Thigh_y_CoM',
                 'L_Up_Arm_x_CoM', 'L_Up_Arm_y_CoM', 'R_Up_Arm_x_CoM', 'R_Up_Arm_y_CoM',
                 'L_Forearm_x_CoM', 'L_Forearm_y_CoM', 'R_Forearm_x_CoM', 'R_Forearm_y_CoM',
                 'Mid_Trunk_x_CoM', 'Mid_Trunk_y_CoM'])
        else:
            OriginalPredictedDiver2DPose.objects.bulk_update(
                diver_2d_pose_bulk_update_list,
                ['diver_x_CoM', 'diver_y_CoM',
                 'Head_CoM', 'Head_CoM',
                 'L_Shank_x_CoM', 'L_Shank_y_CoM', 'R_Shank_x_CoM', 'R_Shank_y_CoM',
                 'L_Thigh_x_CoM', 'L_Thigh_y_CoM', 'R_Thigh_x_CoM', 'R_Thigh_y_CoM',
                 'L_Up_Arm_x_CoM', 'L_Up_Arm_y_CoM', 'R_Up_Arm_x_CoM', 'R_Up_Arm_y_CoM',
                 'L_Forearm_x_CoM', 'L_Forearm_y_CoM', 'R_Forearm_x_CoM', 'R_Forearm_y_CoM',
                 'Mid_Trunk_x_CoM', 'Mid_Trunk_y_CoM'])


def get_CoM_list(video, predicted_diver_2d_pose_list):
    X_LEar_list, Y_LEar_list, X_REar_list, Y_REar_list, \
        X_LSho_list, Y_LSho_list, X_RSho_list, Y_RSho_list, \
        X_LElb_list, Y_LElb_list, X_RElb_list, Y_RElb_list, \
        X_LWri_list, Y_LWri_list, X_RWri_list, Y_RWri_list, \
        X_LHip_list, Y_LHip_list, X_RHip_list, Y_RHip_list, \
        X_LKne_list, Y_LKne_list, X_RKne_list, Y_RKne_list, \
        X_LAnk_list, Y_LAnk_list, X_RAnk_list, Y_RAnk_list, \
        frame_id_list = get_keypoints_list(predicted_diver_2d_pose_list)

    # Get Segment Center of Mass Location
    Head_X_CoM_np, Head_Y_CoM_np, \
        L_Shank_X_CoM_np, L_Shank_Y_CoM_np, R_Shank_X_CoM_np, R_Shank_Y_CoM_np, \
        L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np, R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np, \
        L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np, R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np, \
        L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np, R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np, \
        Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np = \
        get_segment_CoM_location(X_LEar_list, Y_LEar_list, X_REar_list, Y_REar_list,
                                 X_LSho_list, Y_LSho_list, X_RSho_list, Y_RSho_list,
                                 X_LElb_list, Y_LElb_list, X_RElb_list, Y_RElb_list,
                                 X_LWri_list, Y_LWri_list, X_RWri_list, Y_RWri_list,
                                 X_LHip_list, Y_LHip_list, X_RHip_list, Y_RHip_list,
                                 X_LKne_list, Y_LKne_list, X_RKne_list, Y_RKne_list,
                                 X_LAnk_list, Y_LAnk_list, X_RAnk_list, Y_RAnk_list)

    x_body_CoM_np, y_body_CoM_np = get_body_CoM_location(Head_X_CoM_np, Head_Y_CoM_np,
                                                         L_Shank_X_CoM_np, L_Shank_Y_CoM_np,
                                                         R_Shank_X_CoM_np, R_Shank_Y_CoM_np,
                                                         L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np,
                                                         R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np,
                                                         L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np,
                                                         R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np,
                                                         L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np,
                                                         R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np,
                                                         Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np)

    filtered_x_body_CoM_np = get_enhanced_data_list(video, x_body_CoM_np)
    filtered_y_body_CoM_np = get_enhanced_data_list(video, y_body_CoM_np)

    return filtered_x_body_CoM_np, filtered_y_body_CoM_np, \
        Head_X_CoM_np, Head_Y_CoM_np, \
        L_Shank_X_CoM_np, L_Shank_Y_CoM_np, R_Shank_X_CoM_np, R_Shank_Y_CoM_np, \
        L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np, R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np, \
        L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np, R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np, \
        L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np, R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np, \
        Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np


def diver_trajectory_using_CoM(show_video_frame, verbose, video):
    diver_2d_results_folder = 'diver_2d_pose_results'

    # **********************************
    # 1) Get the original diver's 2D poses
    # **********************************
    print("\n1) Get the original diver's 2D poses")
    start_time = time.time()

    original_diver_2d_pose_list = OriginalPredictedDiver2DPose.objects.filter(video=video)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 2) Save output video with the diver tracking using the center of mass of the original diver's 2D pose
    # **********************************
    print("\n2) Save output video with the diver tracking using the center of mass of the original diver's 2D pose")
    start_time = time.time()

    output_video_name_suffix = 'diver_trajectory_using_the_original_diver_2D_pose_CoM'
    output_mp4_video_full_path = \
        save_output_video_with_the_diver_2d_pose_CoM_tracking_line(
            original_diver_2d_pose_list, video, verbose, diver_2d_results_folder, output_video_name_suffix,
            show_video_frame, window_name="Diver tracking using the center of mass of the original diver's 2D pose")

    # Associate video file with video object (original diver's 2D pose)
    video.video_with_the_original_diver_2D_pose_CoM = \
        'videos' + os.sep + diver_2d_results_folder + os.sep + '{}' \
            .format(os.path.basename(output_mp4_video_full_path))
    video.save()

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 3) Get the enhanced diver's 2D poses
    # **********************************
    print("\n3) Get the enhanced diver's 2D poses")
    start_time = time.time()

    enhanced_diver_2d_pose_list = EnhancedPredictedDiver2DPose.objects.filter(video=video)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # **********************************
    # 4) Save output video with the diver tracking using the center of mass of the enhanced diver's 2D pose
    # **********************************
    print("\n4) Save output video with the diver tracking using the center of mass of the enhanced diver's 2D pose")
    start_time = time.time()

    output_video_name_suffix = 'diver_trajectory_using_the_enhanced_diver_2D_pose_CoM'
    output_mp4_video_full_path = \
        save_output_video_with_the_diver_2d_pose_CoM_tracking_line(
            enhanced_diver_2d_pose_list, video, verbose, diver_2d_results_folder, output_video_name_suffix,
            show_video_frame, window_name="Diver tracking using the center of mass of the enhanced diver's 2D pose")

    # Associate video file with video object (enhanced diver's 2D pose)
    video.video_with_the_enhanced_diver_2D_pose_CoM = \
        'videos' + os.sep + diver_2d_results_folder + os.sep + '{}' \
            .format(os.path.basename(output_mp4_video_full_path))
    video.save()

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))


def diver_CoM_velocity_and_acceleration_estimation(video):

    # 1.1) Original diver's 2D pose CoM velocity and acceleration
    print("\n1.1) Original diver's 2D pose CoM velocity and acceleration")
    start_time = time.time()

    diver_original_2d_pose_CoM_velocity_and_acceleration(video)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))

    # 1.2) Enhanced diver's 2D pose CoM velocity and acceleration
    print("\n1.2) Enhanced diver's 2D pose CoM velocity and acceleration")
    start_time = time.time()

    diver_enhanced_2d_pose_CoM_velocity_and_acceleration(video)

    print('Took {0:.2f} seconds\n\n'.format(time.time() - start_time))


def diver_original_2d_pose_CoM_velocity_and_acceleration(video):
    # **********************************
    # 1) Get the original diver's 2D poses
    # **********************************
    original_diver_2d_pose_list = OriginalPredictedDiver2DPose.objects.filter(video=video)

    # **********************************
    # 1.1) Get the original diver's 2D poses velocity lists
    # **********************************
    centred_velocity_list, horizontal_velocity_list, vertical_velocity_list = \
        get_CoM_velocity(original_diver_2d_pose_list, video)

    # **********************************
    # 1.2) Get the original diver's 2D poses acceleration lists
    # **********************************
    centred_acceleration_list, horizontal_acceleration_list, vertical_acceleration_list = \
        get_CoM_acceleration(video, centred_velocity_list, horizontal_velocity_list, vertical_velocity_list)

    # **********************************
    # 1.3) Update the database with the original diver's 2D poses velocity and acceleration
    # **********************************
    save_diver_2d_poses_velocity_and_acceleration \
        (original_diver_2d_pose_list, centred_velocity_list, horizontal_velocity_list, vertical_velocity_list,
         centred_acceleration_list, horizontal_acceleration_list, vertical_acceleration_list,
         is_enhanced_diver_2d_poses=False)


def diver_enhanced_2d_pose_CoM_velocity_and_acceleration(video):
    # **********************************
    # 1) Get the enhanced diver's 2D poses
    # **********************************
    enhanced_diver_2d_pose_list = EnhancedPredictedDiver2DPose.objects.filter(video=video)

    # **********************************
    # 1.1) Get the enhanced diver's 2D poses velocity lists
    # **********************************
    centred_velocity_list, horizontal_velocity_list, vertical_velocity_list = \
        get_CoM_velocity(enhanced_diver_2d_pose_list, video)

    # **********************************
    # 1.2) Get the enhanced diver's 2D poses acceleration lists
    # **********************************
    centred_acceleration_list, horizontal_acceleration_list, vertical_acceleration_list = \
        get_CoM_acceleration(video, centred_velocity_list, horizontal_velocity_list, vertical_velocity_list)

    # **********************************
    # 1.3) Update the database with the enhanced diver's 2D poses velocity and acceleration
    # **********************************
    save_diver_2d_poses_velocity_and_acceleration \
        (enhanced_diver_2d_pose_list, centred_velocity_list, horizontal_velocity_list, vertical_velocity_list,
         centred_acceleration_list, horizontal_acceleration_list, vertical_acceleration_list,
         is_enhanced_diver_2d_poses=True)


def get_CoM_velocity(diver_2d_pose_list, video):
    # Get centred velocity
    centred_velocity_list = get_centred_velocity(video, diver_2d_pose_list)

    # Get horizontal velocity
    horizontal_velocity_list = get_horizontal_velocity(video, diver_2d_pose_list)

    # Get vertical velocity
    vertical_velocity_list = get_vertical_velocity(video, diver_2d_pose_list)

    return centred_velocity_list, horizontal_velocity_list, vertical_velocity_list


def get_CoM_acceleration(video, centred_velocity_list, horizontal_velocity_list, vertical_velocity_list):

    # Get centred acceleration
    centred_acceleration_list = get_centred_acceleration(video, centred_velocity_list)

    # Get horizontal acceleration
    horizontal_acceleration_list = get_horizontal_acceleration(video, horizontal_velocity_list)

    # Get vertical acceleration
    vertical_acceleration_list = get_vertical_acceleration(video, vertical_velocity_list)

    return centred_acceleration_list, horizontal_acceleration_list, vertical_acceleration_list


def get_centred_velocity(video, diver_2d_pose_list):

    if video.camera_calibration:
        # Predicted X and Y CoM values
        pred_xy_CoM_values = []
        for diver_2d_pose in diver_2d_pose_list:
            pred_xy_CoM_values.append((diver_2d_pose.diver_x_CoM, diver_2d_pose.diver_y_CoM))

        centred_velocity_list = get_centred_velocity_list(pred_xy_CoM_values, video, filtered_list=True)

        return centred_velocity_list

    else:
        print('No camera calibration is available. It is not possible to calculate the centred velocity in centimetres.')
        return []


def get_horizontal_velocity(video, diver_2d_pose_list):

    if video.camera_calibration:
        # Predicted CoM X
        x_body_CoM_list = []
        for diver_2d_pose in diver_2d_pose_list:
            x_body_CoM_list.append(diver_2d_pose.diver_x_CoM)

        horizontal_velocity_list = get_horizontal_velocity_list(video, x_body_CoM_list, filtered_list=True)

        return horizontal_velocity_list

    else:
        print('No camera calibration is available. It is not possible to calculate the horizontal velocity in centimetres.')
        return []


def get_vertical_velocity(video, diver_2d_pose_list):
    if video.camera_calibration:
        # Predicted CoM Y
        y_body_CoM_list = []
        for diver_2d_pose in diver_2d_pose_list:
            y_body_CoM_list.append(diver_2d_pose.diver_y_CoM)

        vertical_velocity_list = get_vertical_velocity_list(video, y_body_CoM_list, filtered_list=True)

        return vertical_velocity_list

    else:
        print(
            'No camera calibration is available. It is not possible to calculate the vertical velocity in centimetres.')
        return []


def get_centred_acceleration(video, centred_velocity_list):

    if video.camera_calibration:

        centred_acceleration_list = get_centred_acceleration_list(centred_velocity_list, video, filtered_list=True)

        return centred_acceleration_list

    else:
        print('No camera calibration is available. It is not possible to calculate the centred acceleration in centimetres.')
        return []


def get_horizontal_acceleration(video, horizontal_velocity_list):
    if video.camera_calibration:

        horizontal_acceleration_list = get_horizontal_acceleration_list(horizontal_velocity_list, video,
                                                                        filtered_list=True)

        return horizontal_acceleration_list

    else:
        print(
            'No camera calibration is available. It is not possible to calculate the horizontal acceleration in centimetres.')
        return []


def get_vertical_acceleration(video, vertical_velocity_list):
    if video.camera_calibration:

        filtered_vertical_acceleration_list = get_vertical_acceleration_list(vertical_velocity_list, video,
                                                                             filtered_list=True)

        return filtered_vertical_acceleration_list

    else:
        print(
            'No camera calibration is available. It is not possible to calculate the vertical acceleration in centimetres.')
        return []


def save_diver_2d_poses_velocity_and_acceleration(diver_2d_pose_list, centred_velocity_list, horizontal_velocity_list, vertical_velocity_list,
                                                  centred_acceleration_list, horizontal_acceleration_list, vertical_acceleration_list, is_enhanced_diver_2d_poses):

    diver_2d_pose_bulk_update_list = []

    for diver_2d_pose, centred_velocity, horizontal_velocity, vertical_velocity, \
            centred_acceleration, horizontal_acceleration, vertical_acceleration \
            in zip(diver_2d_pose_list, centred_velocity_list, horizontal_velocity_list, vertical_velocity_list,
                             centred_acceleration_list, horizontal_acceleration_list, vertical_acceleration_list):

        diver_2d_pose.CoM_centred_velocity = centred_velocity
        diver_2d_pose.CoM_centred_velocity_unit = OriginalPredictedDiver2DPose.MOVEMENT_UNIT[0][0]

        diver_2d_pose.CoM_horizontal_velocity = horizontal_velocity
        diver_2d_pose.CoM_horizontal_velocity_unit = OriginalPredictedDiver2DPose.MOVEMENT_UNIT[0][0]

        diver_2d_pose.CoM_vertical_velocity = vertical_velocity
        diver_2d_pose.CoM_vertical_velocity_unit = OriginalPredictedDiver2DPose.MOVEMENT_UNIT[0][0]

        diver_2d_pose.CoM_centred_acceleration = centred_acceleration
        diver_2d_pose.CoM_centred_acceleration_unit = OriginalPredictedDiver2DPose.MOVEMENT_UNIT[1][0]

        diver_2d_pose.CoM_horizontal_acceleration = horizontal_acceleration
        diver_2d_pose.CoM_horizontal_acceleration_unit = OriginalPredictedDiver2DPose.MOVEMENT_UNIT[1][0]

        diver_2d_pose.CoM_vertical_acceleration = vertical_acceleration
        diver_2d_pose.CoM_vertical_acceleration_unit = OriginalPredictedDiver2DPose.MOVEMENT_UNIT[1][0]

        # Update the diver's 2D pose list
        diver_2d_pose_bulk_update_list.append(diver_2d_pose)

    # Store the updated diver 2D poses with CoM
    store_diver_2d_poses_with_velocity_and_acceleration_on_the_database(diver_2d_pose_bulk_update_list, is_enhanced_diver_2d_poses)


def store_diver_2d_poses_with_velocity_and_acceleration_on_the_database(diver_2d_pose_bulk_update_list, is_enhanced_diver_2d_poses):
    if diver_2d_pose_bulk_update_list:
        '''
        This method updates the provided list of objects into the database in an efficient manner
        '''
        if is_enhanced_diver_2d_poses:
            EnhancedPredictedDiver2DPose.objects.bulk_update(
                diver_2d_pose_bulk_update_list,
                ['CoM_centred_velocity', 'CoM_centred_velocity_unit',
                 'CoM_centred_acceleration', 'CoM_centred_acceleration_unit',
                 'CoM_horizontal_velocity', 'CoM_horizontal_velocity_unit',
                 'CoM_horizontal_acceleration', 'CoM_horizontal_acceleration_unit',
                 'CoM_vertical_velocity', 'CoM_vertical_velocity_unit',
                 'CoM_vertical_acceleration', 'CoM_vertical_acceleration_unit'])
        else:
            OriginalPredictedDiver2DPose.objects.bulk_update(
                diver_2d_pose_bulk_update_list,
                ['CoM_centred_velocity', 'CoM_centred_velocity_unit',
                 'CoM_centred_acceleration', 'CoM_centred_acceleration_unit',
                 'CoM_horizontal_velocity', 'CoM_horizontal_velocity_unit',
                 'CoM_horizontal_acceleration', 'CoM_horizontal_acceleration_unit',
                 'CoM_vertical_velocity', 'CoM_vertical_velocity_unit',
                 'CoM_vertical_acceleration', 'CoM_vertical_acceleration_unit'])
