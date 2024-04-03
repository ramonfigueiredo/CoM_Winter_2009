from enum import Enum


# CSV column name
class CSVColumnNameEnum(Enum):
    ##############################
    # Video Information
    ##############################
    FRAME_NAME = 'Frame Name'
    FRAME_ID = 'Frame ID'
    VIDEO_WIDTH = 'Video Width'
    VIDEO_HEIGHT = 'Video Height'

    ##############################
    # Bounding Boxes
    ##############################
    BBOX_CLASS = 'B. Box Class'
    BBOX_XMIN = 'B. Box X Min.'
    BBOX_YMIN = 'B. Box Y Min.'
    BBOX_XMAX = 'B. Box X Max.'
    BBOX_YMAX = 'B. Box Y Max.'
    X_BBOX_CENTER = 'X B. Box Center'
    Y_BBOX_CENTER = 'Y B. Box Center'
    BBOX_PRED_SCORE = 'B. Box Pred. Score'
    BBOX_PRED_SCORE_THRESHOLD = 'B. Box Pred. Score Thresh.'

    BBOX_LABEL = 'B. Box Label'
    BBOX_WIDTH = 'B. Box Width'
    BBOX_HEIGHT = 'B. Box Height'

    ##############################
    # Springboard
    ##############################
    SPRINGBOARD_HEIGHT_IN_PIXEL = 'Springboard Height (Pixels): Y2 - Y1'
    SPRINGBOARD_HEIGHT_IN_CM = 'Springboard Height (cm): Y2 - Y1'
    SPRINGBOARD_HEIGHT_RIGHT_SIDE_POINT_X1 = "X1 (Moving Springboard's Right Side Point)"
    SPRINGBOARD_HEIGHT_RIGHT_SIDE_POINT_Y1 = "Y1 (Moving Springboard's Right Side Point)"
    SPRINGBOARD_HEIGHT_RIGHT_SIDE_POINT_X2 = "X2 (Resting Springboard's Right Side Point)"
    SPRINGBOARD_HEIGHT_RIGHT_SIDE_POINT_Y2 = "Y2 (Resting Springboard's Right Side Point)"
    SPRINGBOARD_RIGHT_SIDE_CENTRED_VELOCITY = "Springboard right side (x, y) velocity ({})"
    SPRINGBOARD_RIGHT_SIDE_CENTRED_ACCELERATION = "Springboard right side (x, y) acceleration ({})"

    SPRINGBOARD_ANGLE_IN_DEGREES = 'Springboard Angle (degrees)'
    SPRINGBOARD_ANGLE_LEFT_SIDE_POINT_X1 = "X1 (Moving Springboard's Left Side Point)"
    SPRINGBOARD_ANGLE_LEFT_SIDE_POINT_Y1 = "Y1 (Moving Springboard's Left Side Point)"
    SPRINGBOARD_ANGLE_RIGHT_SIDE_POINT_X2 = "X2 (Moving Springboard's Right Side Point)"
    SPRINGBOARD_ANGLE_RIGHT_SIDE_POINT_Y2 = "Y2 (Moving Springboard's Right Side Point)"

    ##############################
    # Water Splash
    ##############################
    SPLASH_PERCENTAGE = 'Splash percentage'

    ##############################
    # Kinematic Metrics
    ##############################
    KINEMATIC_METRIC = 'Kinematic Metric'

    ##############################
    # 2D Diver Pose
    ##############################
    PROB_JOINT_DETECTED_LABEL = 'Prob. of {} Detected'

    NOSE = "Nose"
    L_EYE = "L. Eye"
    R_EYE = "R. Eye"
    L_EAR = "L. Ear"
    R_EAR = "R. Ear"
    L_SHOULDER = "L. Shoulder"
    R_SHOULDER = "R. Shoulder"
    L_ELBOW = "L. Elbow"
    R_ELBOW = "R. Elbow"
    L_WRIST = "L. Wrist"
    R_WRIST = "R. Wrist"
    L_HIP = "L. Hip"
    R_HIP = "R. Hip"
    L_KNEE = "L. Knee"
    R_KNEE = "R. Knee"
    L_ANKLE = "L. Ankle"
    R_ANKLE = "R. Ankle"

    # 17 key points
    X_NOSE = 'X Nose'
    Y_NOSE = 'Y Nose'
    INV_Y_NOSE = 'Y Inverted Y Nose (Video Height - Y)'
    Y_INVERTED_NOSE = 'Y Inverted Nose  (Video Height - Y)'
    NOSE_VISIBILITY = 'Prob. of Nose Detected'
    NOSE_VISIBLE = 'Nose Visible?'

    X_LEYE = 'X L. Eye'
    Y_LEYE = 'Y L. Eye'
    INV_Y_LEYE = 'Y Inverted L. Eye (Video Height - Y)'
    Y_INVERTED_LEYE = 'Y Inverted L. Eye (Video Height - Y)'
    LEYE_VISIBILITY = 'Prob. of L. Eye Detected'
    LEYE_VISIBLE = 'L. Eye Visible?'

    X_REYE = 'X R. Eye'
    Y_REYE = 'Y R. Eye'
    INV_Y_REYE = 'Y Inverted R. Eye (Video Height - Y)'
    Y_INVERTED_REYE = 'Y Inverted R. Eye (Video Height - Y)'
    REYE_VISIBILITY = 'Prob. of R. Eye Detected'
    REYE_VISIBLE = 'R. Eye Visible?'

    X_LEAR = 'X L. Ear'
    Y_LEAR = 'Y L. Ear'
    INV_Y_LEAR = 'Y Inverted L. Ear (Video Height - Y)'
    Y_INVERTED_LEAR = 'Y Inverted L. Ear (Video Height - Y)'
    LEAR_VISIBILITY = 'Prob. of L. Ear Detected'
    LEAR_VISIBLE = 'L. Ear Visible?'

    X_REAR = 'X R. Ear'
    Y_REAR = 'Y R. Ear'
    INV_Y_REAR = 'Y Inverted R. Ear (Video Height - Y)'
    Y_INVERTED_REAR = 'Y Inverted R. Ear (Video Height - Y)'
    REAR_VISIBILITY = 'Prob. of R. Ear Detected'
    REAR_VISIBLE = 'R. Ear Visible?'

    X_LSHO = 'X L. Shoulder'
    Y_LSHO = 'Y L. Shoulder'
    INV_Y_LSHO = 'Y Inverted L. Shoulder (Video Height - Y)'
    Y_INVERTED_LSHO = 'Y Inverted L. Shoulder (Video Height - Y)'
    LSHO_VISIBILITY = 'Prob. of L. Shoulder Detected'
    LSHO_VISIBLE = 'L. Shoulder Visible?'

    X_RSHO = 'X R. Shoulder'
    Y_RSHO = 'Y R. Shoulder'
    INV_Y_RSHO = 'Y Inverted R. Shoulder (Video Height - Y)'
    Y_INVERTED_RSHO = 'Y Inverted R. Shoulder (Video Height - Y)'
    RSHO_VISIBILITY = 'Prob. of R. Shoulder Detected'
    RSHO_VISIBLE = 'R. Shoulder Visible?'

    X_LELB = 'X L. Elbow'
    Y_LELB = 'Y L. Elbow'
    INV_Y_LELB = 'Y Inverted L. Elbow (Video Height - Y)'
    Y_INVERTED_LELB = 'Y Inverted L. Elbow (Video Height - Y)'
    LELB_VISIBILITY = 'Prob. of L. Elbow Detected'
    LELB_VISIBLE = 'L. Elbow Visible?'

    X_RELB = 'X R. Elbow'
    Y_RELB = 'Y R. Elbow'
    INV_Y_RELB = 'Y Inverted R. Elbow (Video Height - Y)'
    Y_INVERTED_RELB = 'Y Inverted R. Elbow (Video Height - Y)'
    RELB_Visibility = 'Prob. of R. Elbow Detected'
    RELB_VISIBLE = 'R. Elbow Visible?'

    X_LWRI = 'X L. Wrist'
    Y_LWRI = 'Y L. Wrist'
    INV_Y_LWRI = 'Y Inverted L. Wrist (Video Height - Y)'
    Y_INVERTED_LWRI = 'Y Inverted L. Wrist (Video Height - Y)'
    LWRI_VISIBILITY = 'Prob. of L. Wrist Detected'
    LWRI_VISIBLE = 'L. Wrist Visible?'

    X_RWRI = 'X R. Wrist'
    Y_RWRI = 'Y R. Wrist'
    INV_Y_RWRI = 'Y Inverted R. Wrist (Video Height - Y)'
    Y_INVERTED_RWRI = 'Y Inverted R. Wrist (Video Height - Y)'
    RWRI_VISIBILITY = 'Prob. of R. Wrist Detected'
    RWRI_VISIBLE = 'R. Wrist Visible?'

    X_LHIP = 'X L. Hip'
    Y_LHIP = 'Y L. Hip'
    INV_Y_LHIP = 'Y Inverted L. Hip (Video Height - Y)'
    Y_INVERTED_LHIP = 'Y Inverted L. Hip (Video Height - Y)'
    LHIP_VISIBILITY = 'Prob. of L. Hip Detected'
    LHIP_VISIBLE = 'L. Hip Visible?'

    X_RHIP = 'X R. Hip'
    Y_RHIP = 'Y R. Hip'
    INV_Y_RHIP = 'Y Inverted R. Hip (Video Height - Y)'
    Y_INVERTED_RHIP = 'Y Inverted R. Hip (Video Height - Y)'
    RHIP_VISIBILITY = 'Prob. of R. Hip Detected'
    RHIP_VISIBLE = 'R. Hip Visible?'

    X_LKNE = 'X L. Knee'
    Y_LKNE = 'Y L. Knee'
    INV_Y_LKNE = 'Y Inverted L. Knee (Video Height - Y)'
    Y_INVERTED_LKNE = 'Y Inverted L. Knee (Video Height - Y)'
    LKNE_VISIBILITY = 'Prob. of L. Knee Detected'
    LKNE_VISIBLE = 'L. Knee Visible?'

    X_RKNE = 'X R. Knee'
    Y_RKNE = 'Y R. Knee'
    INV_Y_RKNE = 'Y Inverted R. Knee (Video Height - Y)'
    Y_INVERTED_RKNE = 'Y Inverted R. Knee (Video Height - Y)'
    RKNE_VISIBILITY = 'Prob. of R. Knee Detected'
    RKNE_VISIBLE = 'R. Knee Visible?'

    X_LANK = 'X L. Ankle'
    Y_LANK = 'Y L. Ankle'
    INV_Y_LANK = 'Y Inverted L. Ankle (Video Height - Y)'
    Y_INVERTED_LANK = 'Y Inverted L. Ankle (Video Height - Y)'
    LANK_VISIBILITY = 'Prob. of L. Ankle Detected'
    LANK_VISIBLE = 'L. Ankle Visible?'

    X_RANK = 'X R. Ankle'
    Y_RANK = 'Y R. Ankle'
    INV_Y_RANK = 'Y Inverted R. Ankle (Video Height - Y)'
    Y_INVERTED_RANK = 'Y Inverted R. Ankle (Video Height - Y)'
    RANK_VISIBILITY = 'Prob. of R. Ankle Detected'
    RANK_VISIBLE = 'R. Ankle Visible?'

    # Num. of Detected Key Points
    NUM_KEYPOINTS = 'Num. of Detected Key Points'

    NUM_ANNOTATED_KEYPOINTS = 'Num. of Annotated Key Points'

    ##############################
    # Diver's Angles
    ##############################
    # Segment Angles
    # ------------------------------------
    # Upper arm => LSho_LElb and RSho_RElb
    # Forearm => LElb_LWri and RElb_RWri
    # Thigh => LHip_LKne and RHip_RKne
    # Shank => LKne_LAnk and RKne_RAnk
    # Trunk => LSho_LHip and RSho_RHip
    # ------------------------------------
    LIMB_SEGMENT_ANGLES = 'LIMB-SEGMENT ANGLES'

    ANGLE_LSHO_LELB = 'L. Upper Arm Angle'
    ANGLE_RSHO_RELB = 'R. Upper Arm Angle'

    ANGLE_LELB_LWRI = 'L. Forearm Angle'
    ANGLE_RELB_RWRI = 'R. Forearm Angle'

    ANGLE_LHIP_LKNE = 'L. Thigh Angle'
    ANGLE_RHIP_RKNE = 'R. Thigh Angle'

    ANGLE_LKNE_LANK = 'L. Shank Angle'
    ANGLE_RKNE_RANK = 'R. Shank Angle'

    ANGLE_LSHO_LHIP = 'L. Trunk Angle'
    ANGLE_RSHO_RHIP = 'R. Trunk Angle'

    # ------------------------------------
    # 3-Key Points Angles
    # ------------------------------------
    JOINT_ANGLES_ANGLES = 'JOINT ANGLES ANGLES'

    ANGLE_LSHO_LELB_LWRI = 'L. Elbow Angle'
    ANGLE_RSHO_RELB_RWRI = 'R. Elbow Angle'

    ANGLE_LELB_LSHO_LHIP = 'L. Shoulder Angle'
    ANGLE_RELB_RSHO_RHIP = 'R. Shoulder Angle'

    ANGLE_LSHO_LHIP_LKNE = 'L. Hip Angle'
    ANGLE_RSHO_RHIP_RKNE = 'R. Hip Angle'

    ANGLE_LHIP_LKNE_LANK = 'L. Knee Angle'
    ANGLE_RHIP_RKNE_RANK = 'R. Knee Angle'

    ##############################
    # Diver's Center of Mass (CoM)
    ##############################
    DIVER_X_CM = 'Diver X CoM'
    DIVER_Y_CM = 'Diver Y CoM'
    GENDER = 'Gender'
    HEAD_X_CM = 'Head CoM'
    HEAD_Y_CM = 'Head CoM'
    R_SHANK_X_CM = 'R. Shank X CoM'
    R_SHANK_Y_CM = 'R. Shank Y CoM'
    L_SHANK_X_CM = 'L. Shank X CoM'
    L_SHANK_Y_CM = 'L. Shank Y CoM'
    R_THIGH_X_CM = 'R. Thigh X CoM'
    R_THIGH_Y_CM = 'R. Thigh Y CoM'
    L_THIGH_X_CM = 'L. Thigh X CoM'
    L_THIGH_Y_CM = 'L. Thigh Y CoM'
    R_UP_ARM_X_CM = 'R. Up Arm X CoM'
    R_UP_ARM_Y_CM = 'R. Up Arm Y CoM'
    L_UP_ARM_X_CM = 'L. Up Arm X CoM'
    L_UP_ARM_Y_CM = 'L. Up Arm Y CoM'
    R_FOREARM_X_CM = 'R. Forearm X CoM'
    R_FOREARM_Y_CM = 'R. Forearm Y CoM'
    L_FOREARM_X_CM = 'L. Forearm X CoM'
    L_FOREARM_Y_CM = 'L. Forearm Y CoM'
    MID_TRUNK_X_CM = 'Mid. Trunk X CoM'
    MID_TRUNK_Y_CM = 'Mid. Trunk Y CoM'
    CM_CENTRED_VELOCITY = 'CoM Centred Velocity'
    CM_CENTRED_ACCELERATION = 'CoM Centred Acceleration'
    CM_HORIZONTAL_VELOCITY = 'CoM Horizontal Velocity'
    CM_HORIZONTAL_ACCELERATION = 'CoM Horizontal Acceleration'
    CM_VERTICAL_VELOCITY = 'CoM Vertical Velocity'
    CM_VERTICAL_ACCELERATION = 'CoM Vertical Acceleration'


if __name__ == '__main__':
    for val in CSVColumnNameEnum:
        print(val.value)
