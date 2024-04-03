import math
import os
import time

import cv2
import numpy as np

from core.diver_2d_pose_estimation.diver_2d_pose_estimation_utils import draw_pose, get_pred_joints_list
from core.io.CSVColumnNameEnum import CSVColumnNameEnum
from core.video.frames_utils import get_resized_image
from core.video.frames_utils import show_frame_id, get_video_frame_rotation_angle
from core.video.video_processing.video_processing_utils import create_mp4_video_using_ffmpeg

MEDIA_ROOT = ''


def get_segment_angle_in_degrees(pt1_x, pt1_y, pt2_x, pt2_y, round_value=True, precision_decimal_digits=2):
    """
    Get segment (line) angle in degrees.

    pt1_x: x value of point 1.
    pt1_y: y value of point 1.
    pt2_x: x value of point 2.
    pt2_y: y value of point 2.
    round_value: If true, round a number to a given precision.
    precision_decimal_digits: precision in decimal digits.
    """

    radian_value = get_segment_angle_in_radian(pt1_x, pt1_y, pt2_x, pt2_y, round_value=False)
    degree_value = math.degrees(radian_value)

    return round(degree_value, precision_decimal_digits) if round_value else degree_value


def get_segment_angle_in_radian(pt1_x, pt1_y, pt2_x, pt2_y, round_value=True, precision_decimal_digits=2):
    """
    Get segment (line) in radian.

    pt1_x: x value of point 1.
    pt1_y: y value of point 1.
    pt2_x: x value of point 2.
    pt2_y: y value of point 2.
    round_value: If true, round a number to a given precision.
    precision_decimal_digits: precision in decimal digits.
    """
    radian_value = math.atan2(pt1_y - pt2_y, pt1_x - pt2_x)

    return round(radian_value, precision_decimal_digits) if round_value else radian_value


def get_joint_angle_in_degrees(pt1, pt2, pt3, round_value=True, precision_decimal_digits=2):
    """
    Get joint angle in degree

    pt1: point 1 (x, y)
    pt2: point 2 (x, y)
    pt3: point 3 (x, y)
    """
    pt2_pt1 = pt1 - pt2
    pt2_pt3 = pt3 - pt2

    cosine_angle = np.dot(pt2_pt1, pt2_pt3) / (np.linalg.norm(pt2_pt1) * np.linalg.norm(pt2_pt3))

    angle = np.arccos(cosine_angle)

    angle_in_degrees = np.degrees(angle)

    return round(angle_in_degrees, precision_decimal_digits) if round_value else angle_in_degrees


def save_output_video_with_the_angles_in_the_diver_2D_pose(diver_2d_pose_list, video, verbose,
                                                               diver_2d_results_folder, output_video_name_suffix,
                                                               show_video_frame, window_name='Frame'):
    start_time = time.time()

    video_path = video.uploaded_video_file.url.replace('/media', MEDIA_ROOT)
    if verbose:
        print('\tVideo full path: {}'.format(video_path))
        print('\tsave_output_video_with_the_angles_in_the_diver_2D_pose() started!')

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

                # Draw diver angles
                draw_angles(image_bgr, diver_2d_pose)

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
        print('\tsave_output_video_with_the_angles_in_the_diver_2D_pose() finished! '
              'Took {} seconds'.format(time.time() - start_time))

    return output_mp4_video_full_path


def draw_angles(img, diver_2d_pose):
    # Draw the protractor
    draw_protractor(img)

    ####################################################
    # Segment angle: 2-Key Points Angles
    ####################################################

    limb_segment_angles_y_position = 160

    # Option to include the dark background in the limb segment angles text and animation
    cv2.rectangle(img, (707, limb_segment_angles_y_position - 20), (982, limb_segment_angles_y_position + 230), color=(50, 50, 50), thickness=-1)

    put_angle_text(img, CSVColumnNameEnum.LIMB_SEGMENT_ANGLES.value, angle_point=(742, limb_segment_angles_y_position))

    # pred_L_Upper_arm_angle
    angle_value = diver_2d_pose.pred_L_Upper_arm_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LSHO_LELB.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 40))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 35), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Upper_arm_angle
    angle_value = diver_2d_pose.pred_R_Upper_arm_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RSHO_RELB.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 60))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 55), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_L_Forearm_angle
    angle_value = diver_2d_pose.pred_L_Forearm_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LELB_LWRI.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 80))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 75), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Forearm_angle
    angle_value = diver_2d_pose.pred_R_Forearm_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RELB_RWRI.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 100))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 95), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_L_Thigh_angle
    angle_value = diver_2d_pose.pred_L_Thigh_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LHIP_LKNE.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 120))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 115), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Thigh_angle
    angle_value = diver_2d_pose.pred_R_Thigh_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RHIP_RKNE.value, angle_value)
    #print(angle_text)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 140))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 135), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_L_Shank_angle
    angle_value = diver_2d_pose.pred_L_Shank_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LKNE_LANK.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 160))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 155), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Shank_angle
    angle_value = diver_2d_pose.pred_R_Shank_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RKNE_RANK.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 180))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 175), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_L_Trunk_angle
    angle_value = diver_2d_pose.pred_L_Trunk_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LSHO_LHIP.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 200))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 195), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Trunk_angle
    angle_value = diver_2d_pose.pred_R_Trunk_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RSHO_RHIP.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(742, limb_segment_angles_y_position + 220))
    draw_angle_animation(img, center_point=(727, limb_segment_angles_y_position + 215), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2, counterclockwise=True,
                         show_angle_text=False)

    ####################################################
    # Joint angle: 3-Key Points Angles
    ####################################################

    joint_angles_y_position = 160

    # Option to include the dark background in the joint angles text and animation
    cv2.rectangle(img, (5, joint_angles_y_position - 20), (265, joint_angles_y_position + 190), color=(50, 50, 50), thickness=-1)

    put_angle_text(img, CSVColumnNameEnum.JOINT_ANGLES_ANGLES.value, angle_point=(40, joint_angles_y_position))

    # pred_L_Elbow_angle
    angle_value = diver_2d_pose.pred_L_Elbow_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LSHO_LELB_LWRI.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(40, joint_angles_y_position + 40))
    draw_angle_animation(img, center_point=(25, joint_angles_y_position + 35), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2,
                         counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Elbow_angle
    angle_value = diver_2d_pose.pred_R_Elbow_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RSHO_RELB_RWRI.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(40, joint_angles_y_position + 60))
    draw_angle_animation(img, center_point=(25, joint_angles_y_position + 55), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2,
                         counterclockwise=True,
                         show_angle_text=False)

    # pred_L_Shoulder_angle
    angle_value = diver_2d_pose.pred_L_Shoulder_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LELB_LSHO_LHIP.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(40, joint_angles_y_position + 80))
    draw_angle_animation(img, center_point=(25, joint_angles_y_position + 75), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2,
                         counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Shoulder_angle
    angle_value = diver_2d_pose.pred_R_Shoulder_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RELB_RSHO_RHIP.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(40, joint_angles_y_position + 100))
    draw_angle_animation(img, center_point=(25, joint_angles_y_position + 95), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2,
                         counterclockwise=True,
                         show_angle_text=False)

    # pred_L_Hip_angle
    angle_value = diver_2d_pose.pred_L_Hip_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LSHO_LHIP_LKNE.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(40, joint_angles_y_position + 120))
    draw_angle_animation(img, center_point=(25, joint_angles_y_position + 115), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2,
                         counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Hip_angle
    angle_value = diver_2d_pose.pred_R_Hip_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RSHO_RHIP_RKNE.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(40, joint_angles_y_position + 140))
    draw_angle_animation(img, center_point=(25, joint_angles_y_position + 135), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2,
                         counterclockwise=True,
                         show_angle_text=False)

    # pred_L_Knee_angle
    angle_value = diver_2d_pose.pred_L_Knee_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_LHIP_LKNE_LANK.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(40, joint_angles_y_position + 160))
    draw_angle_animation(img, center_point=(25, joint_angles_y_position + 155), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2,
                         counterclockwise=True,
                         show_angle_text=False)

    # pred_R_Knee_angle
    angle_value = diver_2d_pose.pred_R_Knee_angle
    angle_text = '{}: {}'.format(CSVColumnNameEnum.ANGLE_RHIP_RKNE_RANK.value, angle_value)
    put_angle_text(img, angle_text, angle_point=(40, joint_angles_y_position + 180))
    draw_angle_animation(img, center_point=(25, joint_angles_y_position + 175), ellipse_axes=(8, 8), ellipse_rotation_angle=0,
                         start_angle=0, end_angle=angle_value, thickness=2,
                         counterclockwise=True,
                         show_angle_text=False)


def draw_protractor(img):
    protractor_center_point = (900, 65)
    protractor_axis = (40, 40)

    cv2.putText(img, 'Angles are measured in a', (580, 20), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0),
                thickness=2)
    cv2.putText(img, 'counterclockwise direction', (575, 40), cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                color=(0, 255, 0), thickness=2)
    cv2.ellipse(img, center=protractor_center_point, axes=(40, 40), angle=0, startAngle=0, endAngle=360,
                color=(0, 0, 255), thickness=2)
    cv2.circle(img, protractor_center_point, radius=3, color=(0, 0, 255), thickness=3)

    cv2.arrowedLine(img, (940, 65), (940, 55), color=(0, 255, 0), thickness=2, tipLength=0.5)
    cv2.putText(img, "0", (953, 60), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.putText(img, "360", (945, 80), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=2)

    for angle in range(0, 361, 45):
        x = int(protractor_center_point[0] + protractor_axis[0] * math.cos(angle * math.pi / 180.0))
        y = int(protractor_center_point[1] + protractor_axis[0] * math.sin(angle * math.pi / 180.0))
        cv2.line(img, (900, 65), (x, y), color=(0, 0, 255), thickness=2)

        if angle != 0 and angle != 360:
            angle = 360 - angle

            if angle == 45:
                x_offset = 5
                y_offset = 0

            if angle == 90:
                x_offset = -10
                y_offset = -5

            if angle == 135:
                x_offset = -35
                y_offset = 0

            if angle == 180:
                x_offset = -35
                y_offset = 5

            if angle == 225:
                x_offset = -30
                y_offset = 15

            if angle == 270:
                x_offset = -15
                y_offset = 15

            if angle == 315:
                x_offset = 0
                y_offset = 15

            cv2.putText(img, str(angle), (x + x_offset, y + y_offset), cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                        color=(255, 255, 255), thickness=2)


def put_angle_text(img, angle_text, angle_point):
    cv2.putText(img, angle_text, angle_point, cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=2)


def draw_angle_animation(img, center_point, ellipse_axes, ellipse_rotation_angle, start_angle, end_angle, thickness=3, counterclockwise=False, show_angle_text=False):

    cv2.ellipse(img, center=center_point, axes=ellipse_axes, angle=ellipse_rotation_angle, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=thickness)

    if counterclockwise:
        start_angle = 360 - start_angle
        end_angle = 360 - end_angle

    cv2.ellipse(img, center=center_point, axes=ellipse_axes, angle=ellipse_rotation_angle, startAngle=start_angle, endAngle=end_angle, color=(0, 255, 0), thickness=thickness)

    if show_angle_text:
        # e.g.: 180.12
        if len(str(end_angle)) == 6:
            x_text_offset = 15
        # e.g.: 90.34
        elif len(str(end_angle)) == 5:
            x_text_offset = 10
        # e.g.: 15.56
        elif len(str(end_angle)) == 4:
            x_text_offset = 5
        else:
            x_text_offset = 0

        put_angle_text(img, str(end_angle), (center_point[0] - x_text_offset, center_point[1] + 5))
