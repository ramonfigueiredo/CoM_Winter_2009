import os
import time

import cv2
import numpy as np

from core.diver_2d_pose_estimation.diver_2d_pose_estimation_utils import draw_pose, get_pred_joints_list
from core.video.frames_utils import get_resized_image, show_frame_id, get_video_frame_rotation_angle
from core.video.video_processing.video_processing_utils import create_mp4_video_using_ffmpeg

MEDIA_ROOT = ''

SEGMENT_MASS_PERCENTS = {
    'head': 8.1,
    'trunk': 49.9,
    'upper_arm': 2.8,
    'forearm': 2.2,
    'thigh': 10.0,
    'shank': 6.0

}

SEGMENT_LENGTH_PERCENTS_FROM_PROXIMAL = {
    'trunk': 50,
    'upper_arm': 43.6,
    'forearm': 68.2, # Forearm and hand = Segment length = 68.2 (from proximal) = from Winter's Book, 2009, Table 4.1.
    'thigh': 43.3,
    'shank': 60.6, # Foot and leg = Segment length = 60.6 (from proximal) = from Winter's Book, 2009, Table 4.1.
}


def get_keypoints_list(predicted_diver_2d_pose_list):
    X_LEar_list = []
    Y_LEar_list = []
    X_REar_list = []
    Y_REar_list = []
    X_LSho_list = []
    Y_LSho_list = []
    X_RSho_list = []
    Y_RSho_list = []
    X_LElb_list = []
    Y_LElb_list = []
    X_RElb_list = []
    Y_RElb_list = []
    X_LWri_list = []
    Y_LWri_list = []
    X_RWri_list = []
    Y_RWri_list = []
    X_LHip_list = []
    Y_LHip_list = []
    X_RHip_list = []
    Y_RHip_list = []
    X_LKne_list = []
    Y_LKne_list = []
    X_RKne_list = []
    Y_RKne_list = []
    X_LAnk_list = []
    Y_LAnk_list = []
    X_RAnk_list = []
    Y_RAnk_list = []
    frame_id_list = []

    for diver_2d_pose in predicted_diver_2d_pose_list:
        X_LEar_list.append(diver_2d_pose.pred_x_LEar)
        Y_LEar_list.append(diver_2d_pose.pred_y_LEar)

        X_REar_list.append(diver_2d_pose.pred_x_REar)
        Y_REar_list.append(diver_2d_pose.pred_y_REar)

        X_LSho_list.append(diver_2d_pose.pred_x_LSho)
        Y_LSho_list.append(diver_2d_pose.pred_y_LSho)

        X_RSho_list.append(diver_2d_pose.pred_x_RSho)
        Y_RSho_list.append(diver_2d_pose.pred_y_RSho)

        X_LElb_list.append(diver_2d_pose.pred_x_LElb)
        Y_LElb_list.append(diver_2d_pose.pred_y_LElb)

        X_RElb_list.append(diver_2d_pose.pred_x_RElb)
        Y_RElb_list.append(diver_2d_pose.pred_y_RElb)

        X_LWri_list.append(diver_2d_pose.pred_x_LWri)
        Y_LWri_list.append(diver_2d_pose.pred_y_LWri)

        X_RWri_list.append(diver_2d_pose.pred_x_RWri)
        Y_RWri_list.append(diver_2d_pose.pred_y_RWri)

        X_LHip_list.append(diver_2d_pose.pred_x_LHip)
        Y_LHip_list.append(diver_2d_pose.pred_y_LHip)

        X_RHip_list.append(diver_2d_pose.pred_x_RHip)
        Y_RHip_list.append(diver_2d_pose.pred_y_RHip)

        X_LKne_list.append(diver_2d_pose.pred_x_LKne)
        Y_LKne_list.append(diver_2d_pose.pred_y_LKne)

        X_RKne_list.append(diver_2d_pose.pred_x_RKne)
        Y_RKne_list.append(diver_2d_pose.pred_y_RKne)

        X_LAnk_list.append(diver_2d_pose.pred_x_LAnk)
        Y_LAnk_list.append(diver_2d_pose.pred_y_LAnk)

        X_RAnk_list.append(diver_2d_pose.pred_x_RAnk)
        Y_RAnk_list.append(diver_2d_pose.pred_y_RAnk)

        frame_id_list.append(diver_2d_pose.frame_id)


    return X_LEar_list, Y_LEar_list, X_REar_list, Y_REar_list, \
        X_LSho_list, Y_LSho_list, X_RSho_list, Y_RSho_list, \
        X_LElb_list, Y_LElb_list, X_RElb_list, Y_RElb_list, \
        X_LWri_list, Y_LWri_list, X_RWri_list, Y_RWri_list, \
        X_LHip_list, Y_LHip_list, X_RHip_list, Y_RHip_list, \
        X_LKne_list, Y_LKne_list, X_RKne_list, Y_RKne_list, \
        X_LAnk_list, Y_LAnk_list, X_RAnk_list, Y_RAnk_list, \
        frame_id_list


def get_segment_CoM_location(X_LEar_list, Y_LEar_list, X_REar_list, Y_REar_list,
                             X_LSho_list, Y_LSho_list, X_RSho_list, Y_RSho_list,
                             X_LElb_list, Y_LElb_list, X_RElb_list, Y_RElb_list,
                             X_LWri_list, Y_LWri_list, X_RWri_list, Y_RWri_list,
                             X_LHip_list, Y_LHip_list, X_RHip_list, Y_RHip_list,
                             X_LKne_list, Y_LKne_list, X_RKne_list, Y_RKne_list,
                             X_LAnk_list, Y_LAnk_list, X_RAnk_list, Y_RAnk_list):

    # Head = LEar and REar
    Head_X_CoM_np, Head_Y_CoM_np = \
        get_segment_CoM('head', X_LEar_list, Y_LEar_list, X_REar_list, Y_REar_list)

    # Left Shank = LKne and LAnk
    L_Shank_X_CoM_np, L_Shank_Y_CoM_np = \
        get_segment_CoM('shank', X_LKne_list, Y_LKne_list, X_LAnk_list, Y_LAnk_list)

    # Right Shank = RKne and RAnk
    R_Shank_X_CoM_np, R_Shank_Y_CoM_np = \
        get_segment_CoM('shank', X_RKne_list, Y_RKne_list, X_RAnk_list, Y_RAnk_list)

    # Left Thigh = LHip and LKne
    L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np = \
        get_segment_CoM('thigh', X_LHip_list, Y_LHip_list, X_LKne_list, Y_LKne_list)

    # Right Thigh = RHip and RKne
    R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np = \
        get_segment_CoM('thigh', X_RHip_list, Y_RHip_list, X_RKne_list, Y_RKne_list)

    # Left Up Arm = LSho and LElb
    L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np = \
        get_segment_CoM('upper_arm', X_LSho_list, Y_LSho_list, X_LElb_list, Y_LElb_list)

    # Right Up Arm = RSho and RElb
    R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np = \
        get_segment_CoM('upper_arm', X_RSho_list, Y_RSho_list, X_RElb_list, Y_RElb_list)

    # Left Forearm = LElb and LWri
    L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np = \
        get_segment_CoM('forearm', X_LElb_list, Y_LElb_list, X_LWri_list, Y_LWri_list)

    # Right Forearm = RElb and RWri
    R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np = \
        get_segment_CoM('forearm', X_RElb_list, Y_RElb_list, X_RWri_list, Y_RWri_list)

    # Mid Trunk: The trunk segment is defined as running from the midpoints between hips and shoulders
    X_LHip_LSho = [((x1 + x2) / 2) for x1, x2 in zip(X_LHip_list, X_LSho_list)]
    Y_LHip_LSho = [((y1 + y2) / 2) for y1, y2 in zip(Y_LHip_list, Y_LSho_list)]

    X_RHip_RSho = [((x1 + x2) / 2) for x1, x2 in zip(X_RHip_list, X_RSho_list)]
    Y_RHip_RSho = [((y1 + y2) / 2) for y1, y2 in zip(Y_RHip_list, Y_RSho_list)]

    Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np = \
        get_segment_CoM('trunk', X_LHip_LSho, Y_LHip_LSho, X_RHip_RSho, Y_RHip_RSho)

    return Head_X_CoM_np, Head_Y_CoM_np, \
        L_Shank_X_CoM_np, L_Shank_Y_CoM_np, R_Shank_X_CoM_np, R_Shank_Y_CoM_np, \
        L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np, R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np, \
        L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np, R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np, \
        L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np, R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np, \
        Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np


def get_segment_CoM(segment_name, x_proximal_list, y_proximal_list, x_distal_list, y_distal_list):
    segment_x_CoM_list = []
    segment_y_CoM_list = []

    for x_proximal, y_proximal, x_distal, y_distal in zip(x_proximal_list, y_proximal_list, x_distal_list, y_distal_list):
        if segment_name == 'head':
            segment_cm_x_position = (x_proximal + x_distal) / 2
            segment_cm_y_position = (y_proximal + y_distal) / 2
        else:
            length_percent = SEGMENT_LENGTH_PERCENTS_FROM_PROXIMAL[segment_name] / 100

            segment_cm_x_position = x_proximal + (length_percent * (x_distal - x_proximal))
            segment_cm_y_position = y_proximal + (length_percent * (y_distal - y_proximal))

        segment_x_CoM_list.append(segment_cm_x_position)
        segment_y_CoM_list.append(segment_cm_y_position)

    return np.array(segment_x_CoM_list), np.array(segment_y_CoM_list)


def get_body_CoM_location(Head_X_CoM_np, Head_Y_CoM_np,
                          L_Shank_X_CoM_np, L_Shank_Y_CoM_np, R_Shank_X_CoM_np, R_Shank_Y_CoM_np,
                          L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np, R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np,
                          L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np, R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np,
                          L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np, R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np,
                          Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np):

    # Head = LEar and REar
    Head_X_body_CoM_np, Head_Y_body_CoM_np = \
        get_body_CoM('head', Head_X_CoM_np, Head_Y_CoM_np)

    # Left Shank = LKne and LAnk
    L_Shank_X_body_CoM_np, L_Shank_Y_body_CoM_np = \
        get_body_CoM('shank', L_Shank_X_CoM_np, L_Shank_Y_CoM_np)

    # Right Shank = RKne and RAnk
    R_Shank_X_body_CoM_np, R_Shank_Y_body_CoM_np = \
        get_body_CoM('shank', R_Shank_X_CoM_np, R_Shank_Y_CoM_np)

    # Left Thigh = LHip and LKne
    L_Thigh_X_body_CoM_np, L_Thigh_Y_body_CoM_np = \
        get_body_CoM('thigh', L_Thigh_X_CoM_np, L_Thigh_Y_CoM_np)

    # Right Thigh = RHip and RKne
    R_Thigh_X_body_CoM_np, R_Thigh_Y_body_CoM_np = \
        get_body_CoM('thigh', R_Thigh_X_CoM_np, R_Thigh_Y_CoM_np)

    # Left Up Arm = LSho and LElb
    L_Up_Arm_X_body_CoM_np, L_Up_Arm_Y_body_CoM_np = \
        get_body_CoM('upper_arm', L_Up_Arm_X_CoM_np, L_Up_Arm_Y_CoM_np)

    # Right Up Arm = RSho and RElb
    R_Up_Arm_X_body_CoM_np, R_Up_Arm_Y_body_CoM_np = \
        get_body_CoM('upper_arm', R_Up_Arm_X_CoM_np, R_Up_Arm_Y_CoM_np)

    # Left Forearm = LElb and LWri
    L_Forearm_X_body_CoM_np, L_Forearm_Y_body_CoM_np = \
        get_body_CoM('forearm', L_Forearm_X_CoM_np, L_Forearm_Y_CoM_np)

    # Right Forearm = RElb and RWri
    R_Forearm_X_body_CoM_np, R_Forearm_Y_body_CoM_np = \
        get_body_CoM('forearm', R_Forearm_X_CoM_np, R_Forearm_Y_CoM_np)

    Mid_Trunk_X_body_CoM_np, Mid_Trunk_Y_body_CoM_np = \
        get_body_CoM('trunk', Mid_Trunk_X_CoM_np, Mid_Trunk_Y_CoM_np)

    x_body_CoM_np = (Head_X_body_CoM_np +
                     R_Shank_X_body_CoM_np + L_Shank_X_body_CoM_np +
                     R_Thigh_X_body_CoM_np + L_Thigh_X_body_CoM_np +
                     R_Up_Arm_X_body_CoM_np + L_Up_Arm_X_body_CoM_np +
                     R_Forearm_X_body_CoM_np + L_Forearm_X_body_CoM_np +
                     Mid_Trunk_X_body_CoM_np)

    y_body_CoM_np = (Head_Y_body_CoM_np +
                     R_Shank_Y_body_CoM_np + L_Shank_Y_body_CoM_np +
                     R_Thigh_Y_body_CoM_np + L_Thigh_Y_body_CoM_np +
                     R_Up_Arm_Y_body_CoM_np + L_Up_Arm_Y_body_CoM_np +
                     R_Forearm_Y_body_CoM_np + L_Forearm_Y_body_CoM_np +
                     Mid_Trunk_Y_body_CoM_np)

    return x_body_CoM_np, y_body_CoM_np


def get_body_CoM(segment_name, segment_x_cm_np, segment_y_cm_np):
    body_x_CoM_list = []
    body_y_CoM_list = []

    for segment_x_cm, segment_y_cm in zip(segment_x_cm_np, segment_y_cm_np):
        mass_percent = SEGMENT_MASS_PERCENTS[segment_name] / 100

        CoM_x = segment_x_cm * mass_percent
        CoM_y = segment_y_cm * mass_percent

        body_x_CoM_list.append(CoM_x)
        body_y_CoM_list.append(CoM_y)

    return np.array(body_x_CoM_list), np.array(body_y_CoM_list)


def save_output_video_with_the_diver_2d_pose_CoM_tracking_line(diver_2d_pose_list, video, verbose,
                                                               diver_2d_results_folder, output_video_name_suffix,
                                                               show_video_frame, window_name='Frame'):
    start_time = time.time()

    video_path = video.uploaded_video_file.url.replace('/media', MEDIA_ROOT)
    if verbose:
        print('\tVideo full path: {}'.format(video_path))
        print('\tsave_output_video_with_the_diver_2d_pose_CoM_tracking_line() started!')

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        if verbose:
            print("Error opening video stream or file")

    # Output video

    # Output video path
    output_video_folder = os.path.join(os.path.dirname(video.video_file.path), diver_2d_results_folder)

    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

    output_video_name = os.path.basename(video.name) + '_' + output_video_name_suffix
    output_video_extension = '.avi'

    output_avi_video_full_path = os.path.join(output_video_folder, output_video_name + output_video_extension)

    # Get video frame rotation angle
    frame_rotation_angle = get_video_frame_rotation_angle(video)

    if video.width > video.height:
        video_width, video_height = video.height, video.width
    else:
        video_width, video_height = video.width, video.height

    # Output video object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_avi_video_full_path, fourcc, video.fps, (video_width, video_height))

    # Get center of mass lists (Xs and Ys)
    x_body_CoM_list = []
    y_body_CoM_list = []
    for diver_2d_pose in diver_2d_pose_list:
        x_body_CoM_list.append(diver_2d_pose.diver_x_CoM)
        y_body_CoM_list.append(diver_2d_pose.diver_y_CoM)

    frame_id = 1
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, image_bgr = cap.read()
        if ret:

            if video.uploaded_video_rotation_option != None and frame_rotation_angle != None:
                # Rotate frame
                image_bgr = cv2.rotate(image_bgr, frame_rotation_angle)

            if show_video_frame:
                # Show frame ID
                show_frame_id(image_bgr, frame_id)

            diver_2d_pose_query_set = \
                diver_2d_pose_list.filter(frame_id=frame_id)

            if diver_2d_pose_query_set:
                diver_2d_pose = diver_2d_pose_query_set[0]

                # Return a Numpy array list with shape (17, 2) and the joints (x,y) values in the COCO_KEYPOINT_INDEXES
                # positions
                pred_joints_list = get_pred_joints_list(diver_2d_pose)

                # draw the poses
                draw_pose(pred_joints_list, image_bgr, show_pts_labels=False)
            else:
                diver_2d_pose = None

            # Draw diver center of mass
            draw_diver_CoM(image_bgr, x_body_CoM_list, y_body_CoM_list, diver_2d_pose)

            if show_video_frame:
                # Display the resulting frame
                frame_resized = get_resized_image(image_bgr, scale_percent=60)
                cv2.imshow(window_name, frame_resized)

            # Save output video frame
            output_video.write(image_bgr)

            frame_id = frame_id + 1

            if show_video_frame:
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    if verbose:
        print('\tNumber of frames: {}'.format(frame_id))

    # Closes all the frames
    cv2.destroyAllWindows()

    # When everything done, release the video capture object
    cap.release()

    # release the output video object
    output_video.release()

    # Convert avi to mp4
    output_mp4_video_full_path = output_avi_video_full_path.replace('.avi', '.mp4')
    create_mp4_video_using_ffmpeg(input_video_path=output_avi_video_full_path,
                                  output_video_path=output_mp4_video_full_path,
                                  rotation_option=None)

    if os.path.exists(output_avi_video_full_path):
        # Delete avi video
        os.remove(output_avi_video_full_path)

    if verbose:
        print('\tsave_output_video_with_the_diver_2d_pose_CoM_tracking_line() finished! '
              'Took {} seconds'.format(time.time() - start_time))

    return output_mp4_video_full_path


def draw_diver_CoM(img, x_body_CoM_list, y_body_CoM_list, diver_2d_pose, show_diver_com_with_x_y_values=True):

    # Draw diver's center of mass tracking line
    cm_x_prev, cm_y_prev = x_body_CoM_list[0], y_body_CoM_list[0]
    for cm_x, cm_y in zip(x_body_CoM_list, y_body_CoM_list):
        cv2.line(img, (int(cm_x_prev), int(cm_y_prev)), (int(cm_x), int(cm_y)), (0, 255, 255), 3)
        cm_x_prev, cm_y_prev = cm_x, cm_y

    if diver_2d_pose:
        # Get the diver' center of masses for the video
        x_diver_CoM = diver_2d_pose.diver_x_CoM
        y_diver_CoM = diver_2d_pose.diver_y_CoM

        cv2.circle(img, (int(x_diver_CoM), int(y_diver_CoM)), 10, (0, 255, 255), -1)

        pt_offset = 10
        if show_diver_com_with_x_y_values:
            cm_label = 'CoM ({}, {})'.format(int(x_diver_CoM), int(y_diver_CoM))
        else:
            cm_label = 'CoM'

        cv2.putText(img, cm_label, (int(x_diver_CoM + pt_offset), int(y_diver_CoM + pt_offset)), cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2, color=(255, 255, 255), thickness=3)
