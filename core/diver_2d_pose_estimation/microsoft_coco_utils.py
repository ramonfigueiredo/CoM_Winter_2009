from enum import Enum


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


DIVINGFY_COCO_KEYPOINT_LABELS = [
        "Nose",         # "nose"
        "LEye",         # "left_eye"
        "REye",         # "right_eye"
        "LEar",         # "left_ear"
        "REar",         # "right_ear"
        "LSho",         # "left_shoulder"
        "RSho",         # "right_shoulder"
        "LElb",         # "left_elbow"
        "RElb",         # "right_elbow"
        "LWri",         # "left_wrist"
        "RWri",         # "right_wrist"
        "LHip",         # "left_hip"
        "RHip",         # "right_hip"
        "LKne",         # "left_knee"
        "RKne",         # "right_knee"
        "LAnk",         # "left_ankle"
        "RAnk"          # "right_ankle"
]


class DivingFyCOCOKeypointLabel(Enum):
    Nose = DIVINGFY_COCO_KEYPOINT_LABELS[0]
    LEye = DIVINGFY_COCO_KEYPOINT_LABELS[1]
    REye = DIVINGFY_COCO_KEYPOINT_LABELS[2]
    LEar = DIVINGFY_COCO_KEYPOINT_LABELS[3]
    REar = DIVINGFY_COCO_KEYPOINT_LABELS[4]
    LSho = DIVINGFY_COCO_KEYPOINT_LABELS[5]
    RSho = DIVINGFY_COCO_KEYPOINT_LABELS[6]
    LElb = DIVINGFY_COCO_KEYPOINT_LABELS[7]
    RElb = DIVINGFY_COCO_KEYPOINT_LABELS[8]
    LWri = DIVINGFY_COCO_KEYPOINT_LABELS[9]
    RWri = DIVINGFY_COCO_KEYPOINT_LABELS[10]
    LHip = DIVINGFY_COCO_KEYPOINT_LABELS[11]
    RHip = DIVINGFY_COCO_KEYPOINT_LABELS[12]
    LKne = DIVINGFY_COCO_KEYPOINT_LABELS[13]
    RKne = DIVINGFY_COCO_KEYPOINT_LABELS[14]
    LAnk = DIVINGFY_COCO_KEYPOINT_LABELS[15]
    RAnk = DIVINGFY_COCO_KEYPOINT_LABELS[16]


SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8],
    [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]


CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


NUM_KPTS = 17
