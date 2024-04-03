import logging
import os

import pandas as pd

from core.io.CSVColumnNameEnum import CSVColumnNameEnum
from diving_analysis.models import GroundTruthDiver2DPose


def read_ground_truth_diver_2d_poses(video_name, ground_truth_diver_2d_poses_csv_file_path, verbose):
    ground_truth_diver_2d_poses_list = []

    try:
        if os.path.exists(ground_truth_diver_2d_poses_csv_file_path):
            gt_df = pd.read_csv(ground_truth_diver_2d_poses_csv_file_path)
            gt_df = gt_df.fillna(0)

            GT_PREFIX = 'GT'
            YES_CHAR = 'Y'
            NO_CHAR = 'N'
            YES = 'YES'

            for index, row in gt_df.iterrows():

                ground_truth_diver_2d_pose = GroundTruthDiver2DPose(
                    video_name=video_name,
                    frame_name=row[CSVColumnNameEnum.FRAME_NAME.value],
                    frame_id=row[CSVColumnNameEnum.FRAME_ID.value],

                    ###################################################################################################################
                    # Joints (X, Y) and Probability of Joint Detection
                    ###################################################################################################################

                    gt_x_Nose=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_NOSE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_NOSE.value] != 0 else None,
                    gt_y_Nose=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_NOSE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_NOSE.value] != 0 else None,
                    Nose_visible=YES_CHAR if row[CSVColumnNameEnum.NOSE_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_LEye=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LEYE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LEYE.value] != 0 else None,
                    gt_y_LEye=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LEYE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LEYE.value] != 0 else None,
                    LEye_visible=YES_CHAR if row[CSVColumnNameEnum.LEYE_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_REye=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_REYE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_REYE.value] != 0 else None,
                    gt_y_REye=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_REYE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_REYE.value] != 0 else None,
                    REye_visible=YES_CHAR if row[CSVColumnNameEnum.REYE_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_LEar=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LEAR.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LEAR.value] != 0 else None,
                    gt_y_LEar=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LEAR.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LEAR.value] != 0 else None,
                    LEar_visible=YES_CHAR if row[CSVColumnNameEnum.LEAR_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_REar=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_REAR.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_REAR.value] != 0 else None,
                    gt_y_REar=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_REAR.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_REAR.value] != 0 else None,
                    REar_visible=YES_CHAR if row[CSVColumnNameEnum.REAR_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_LSho=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LSHO.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LSHO.value] != 0 else None,
                    gt_y_LSho=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LSHO.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LSHO.value] != 0 else None,
                    LSho_visible=YES_CHAR if row[CSVColumnNameEnum.LSHO_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_RSho=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RSHO.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RSHO.value] != 0 else None,
                    gt_y_RSho=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RSHO.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RSHO.value] != 0 else None,
                    RSho_visible=YES_CHAR if row[CSVColumnNameEnum.RSHO_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_LElb=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LELB.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LELB.value] != 0 else None,
                    gt_y_LElb=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LELB.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LELB.value] != 0 else None,
                    LElb_visible=YES_CHAR if row[CSVColumnNameEnum.LELB_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_RElb=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RELB.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RELB.value] != 0 else None,
                    gt_y_RElb=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RELB.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RELB.value] != 0 else None,
                    RElb_visible=YES_CHAR if row[CSVColumnNameEnum.RELB_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_LWri=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LWRI.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LWRI.value] != 0 else None,
                    gt_y_LWri=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LWRI.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LWRI.value] != 0 else None,
                    LWri_visible=YES_CHAR if row[CSVColumnNameEnum.LWRI_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_RWri=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RWRI.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RWRI.value] != 0 else None,
                    gt_y_RWri=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RWRI.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RWRI.value] != 0 else None,
                    RWri_visible=YES_CHAR if row[CSVColumnNameEnum.RWRI_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_LHip=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LHIP.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LHIP.value] != 0 else None,
                    gt_y_LHip=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LHIP.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LHIP.value] != 0 else None,
                    LHip_visible=YES_CHAR if row[CSVColumnNameEnum.LHIP_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_RHip=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RHIP.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RHIP.value] != 0 else None,
                    gt_y_RHip=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RHIP.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RHIP.value] != 0 else None,
                    RHip_visible=YES_CHAR if row[CSVColumnNameEnum.RHIP_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_LKne=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LKNE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LKNE.value] != 0 else None,
                    gt_y_LKne=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LKNE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LKNE.value] != 0 else None,
                    LKne_visible=YES_CHAR if row[CSVColumnNameEnum.LKNE_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_RKne=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RKNE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RKNE.value] != 0 else None,
                    gt_y_RKne=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RKNE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RKNE.value] != 0 else None,
                    RKne_visible=YES_CHAR if row[CSVColumnNameEnum.RKNE_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_LAnk=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LANK.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_LANK.value] != 0 else None,
                    gt_y_LAnk=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LANK.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_LANK.value] != 0 else None,
                    LAnk_visible=YES_CHAR if row[CSVColumnNameEnum.LANK_VISIBLE.value].upper() == YES else NO_CHAR,

                    gt_x_RAnk=row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RANK.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.X_RANK.value] != 0 else None,
                    gt_y_RAnk=row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RANK.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.Y_RANK.value] != 0 else None,
                    RAnk_visible=YES_CHAR if row[CSVColumnNameEnum.RANK_VISIBLE.value].upper() == YES else NO_CHAR,

                    ###################################################################################################################
                    # Number of Annotated Key Points (Joints)
                    ###################################################################################################################

                    num_annotated_key_points=row[CSVColumnNameEnum.NUM_ANNOTATED_KEYPOINTS.value],

                    ###################################################################################################################
                    # Segment's Angles: L/R Upper Arm, L/R Forearm, L/R Thigh, L/R Shank, and L/R Trunk
                    ###################################################################################################################

                    gt_L_Upper_arm_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LSHO_LELB.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LSHO_LELB.value] != 0 else None,
                    gt_R_Upper_arm_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RSHO_RELB.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RSHO_RELB.value] != 0 else None,

                    gt_L_Forearm_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LELB_LWRI.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LELB_LWRI.value] != 0 else None,
                    gt_R_Forearm_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RELB_RWRI.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RELB_RWRI.value] != 0 else None,

                    gt_L_Thigh_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LHIP_LKNE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LHIP_LKNE.value] != 0 else None,
                    gt_R_Thigh_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RHIP_RKNE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RHIP_RKNE.value] != 0 else None,

                    gt_L_Shank_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LKNE_LANK.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LKNE_LANK.value] != 0 else None,
                    gt_R_Shank_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RKNE_RANK.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RKNE_RANK.value] != 0 else None,

                    gt_L_Trunk_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LSHO_LHIP.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LSHO_LHIP.value] != 0 else None,
                    gt_R_Trunk_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RSHO_RHIP.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RSHO_RHIP.value] != 0 else None,

                    ###################################################################################################################
                    # Joint's Angles: L/R Elbow, L/R Shoulder, L/R Hip, and L/R Knee
                    ###################################################################################################################

                    gt_L_Elbow_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LSHO_LELB_LWRI.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LSHO_LELB_LWRI.value] != 0 else None,
                    gt_R_Elbow_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RSHO_RELB_RWRI.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RSHO_RELB_RWRI.value] != 0 else None,

                    gt_L_Shoulder_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LELB_LSHO_LHIP.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LELB_LSHO_LHIP.value] != 0 else None,
                    gt_R_Shoulder_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RELB_RSHO_RHIP.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RELB_RSHO_RHIP.value] != 0 else None,

                    gt_L_Hip_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LSHO_LHIP_LKNE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LSHO_LHIP_LKNE.value] != 0 else None,
                    gt_R_Hip_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RSHO_RHIP_RKNE.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RSHO_RHIP_RKNE.value] != 0 else None,

                    gt_L_Knee_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LHIP_LKNE_LANK.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_LHIP_LKNE_LANK.value] != 0 else None,
                    gt_R_Knee_angle=row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RHIP_RKNE_RANK.value] if row[GT_PREFIX + ' ' + CSVColumnNameEnum.ANGLE_RHIP_RKNE_RANK.value] != 0 else None
                )
                ground_truth_diver_2d_poses_list.append(ground_truth_diver_2d_pose)

            print()
        else:
            logging.error("Ground truth diver's 2D poses CSV file {} doesn't exists".format(ground_truth_diver_2d_poses_csv_file_path))
    except:
        logging.error("Error reading the ground truth diver's 2D poses CSV file: {}"
                      .format(ground_truth_diver_2d_poses_csv_file_path))

    if verbose:
        print("The CSV file {} has {} ground truth diver's 2D poses"
              .format(os.path.basename(ground_truth_diver_2d_poses_csv_file_path), len(ground_truth_diver_2d_poses_list)))

    return ground_truth_diver_2d_poses_list
