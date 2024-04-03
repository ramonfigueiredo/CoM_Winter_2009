from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils.translation import gettext_lazy as _

from datasets.models import Video


class OriginalPredictedBoundingBox(models.Model):
    BOX_CLASS = (
        ('1', 'Diver'),
        ('2', 'Springboard'),
        ('3', 'Water splash')
    )

    frame_name = models.CharField(null=True, blank=True, max_length=255)
    frame_id = models.PositiveIntegerField(null=True, blank=True)
    video = models.ForeignKey(Video, null=True, blank=True, on_delete=models.CASCADE)
    bbox_class = models.CharField(null=True, blank=True, max_length=1, choices=BOX_CLASS)
    predicted_score = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                        null=True, blank=True)
    x_min = models.PositiveIntegerField(null=True, blank=True)
    x_max = models.PositiveIntegerField(null=True, blank=True)
    y_min = models.PositiveIntegerField(null=True, blank=True)
    y_max = models.PositiveIntegerField(null=True, blank=True)
    x_bbox_center = models.PositiveIntegerField(null=True, blank=True)
    y_bbox_center = models.PositiveIntegerField(null=True, blank=True)
    bbox_width = models.PositiveIntegerField(null=True, blank=True)
    bbox_height = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ['frame_name']
        verbose_name = "Original predicted bounding box"
        verbose_name_plural = "Original predicted bounding boxes"

    def __str__(self):
        return 'Original predicted bounding box: ' \
               'Video={}, Frame Name={}, Frame ID={} => (' \
               'bbox_class={}, predicted_score={}, ' \
               'xmin={}, xmax={}, ymin={}, ymax={}, ' \
               'x_bbox_center={}, y_bbox_center={}, ' \
               'bbox_width={}, bbox_height={}' \
               ')' \
            .format(self.video.name if self.video else '', self.frame_name, self.frame_id,
                    self.get_bbox_class_display(), self.predicted_score,
                    self.x_min, self.x_max, self.y_min, self.y_max,
                    self.x_bbox_center, self.y_bbox_center,
                    self.bbox_width, self.bbox_height)


class EnhancedPredictedBoundingBox(models.Model):
    frame_name = models.CharField(null=True, blank=True, max_length=255)
    frame_id = models.PositiveIntegerField(null=True, blank=True)
    video = models.ForeignKey(Video, null=True, blank=True, on_delete=models.CASCADE)
    ENHANCED_BOX_CLASS = (
        ('1', 'Diver'),
        ('3', 'Water splash'),
        ('4', 'Top springboard'),
        ('5', 'Bottom springboard')
    )
    bbox_class = models.CharField(null=True, blank=True, max_length=1, choices=ENHANCED_BOX_CLASS)
    predicted_score = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                        null=True, blank=True)
    x_min = models.PositiveIntegerField(null=True, blank=True)
    x_max = models.PositiveIntegerField(null=True, blank=True)
    y_min = models.PositiveIntegerField(null=True, blank=True)
    y_max = models.PositiveIntegerField(null=True, blank=True)
    x_bbox_center = models.PositiveIntegerField(null=True, blank=True)
    y_bbox_center = models.PositiveIntegerField(null=True, blank=True)
    bbox_width = models.PositiveIntegerField(null=True, blank=True)
    bbox_height = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ['frame_name']
        verbose_name = "Enhanced predicted bounding box"
        verbose_name_plural = "Enhanced predicted bounding boxes"

    def __str__(self):
        return 'Enhanced predicted bounding box: ' \
               'Video={}, Frame Name={}, Frame ID={} => (' \
               'bbox_class={}, predicted_score={}, ' \
               'xmin={}, xmax={}, ymin={}, ymax={}, ' \
               'x_bbox_center={}, y_bbox_center={}, ' \
               'bbox_width={}, bbox_height={}' \
               ')' \
            .format(self.video.name if self.video else '', self.frame_name, self.frame_id,
                    self.get_bbox_class_display(), self.predicted_score,
                    self.x_min, self.x_max, self.y_min, self.y_max,
                    self.x_bbox_center, self.y_bbox_center,
                    self.bbox_width, self.bbox_height)


class OriginalPredictedDiver2DPose(models.Model):
    frame_name = models.CharField(null=True, blank=True, max_length=255)
    frame_id = models.PositiveIntegerField(null=True, blank=True)
    video = models.ForeignKey(Video, null=True, blank=True, on_delete=models.CASCADE)

    ###################################################################################################################
    # Joints (X, Y) and Probability of Joint Detection
    ###################################################################################################################

    pred_x_Nose = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_Nose = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_Nose_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_LEye = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LEye = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LEye_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_REye = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_REye = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_REye_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_LEar = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LEar = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LEar_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_REar = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_REar = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_REar_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_LSho = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LSho = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LSho_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_RSho = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RSho = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RSho_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_LElb = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LElb = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LElb_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_RElb = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RElb = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RElb_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_LWri = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LWri = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LWri_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_RWri = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RWri = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RWri_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_LHip = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LHip = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LHip_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_RHip = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RHip = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RHip_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_LKne = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LKne = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LKne_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_RKne = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RKne = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RKne_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_LAnk = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LAnk = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LAnk_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    pred_x_RAnk = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RAnk = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RAnk_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)

    ###################################################################################################################
    # Number of Detected Key Points (Joints)
    ###################################################################################################################

    num_detected_key_points = models.PositiveIntegerField(null=True, blank=True)

    ###################################################################################################################
    # Segment's Angles: L/R Upper Arm, L/R Forearm, L/R Thigh, L/R Shank, and L/R Trunk
    ###################################################################################################################

    pred_L_Upper_arm_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                               null=True, blank=True)
    pred_R_Upper_arm_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                               null=True, blank=True)

    pred_L_Forearm_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                             null=True, blank=True)
    pred_R_Forearm_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                             null=True, blank=True)

    pred_L_Thigh_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)
    pred_R_Thigh_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)

    pred_L_Shank_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)
    pred_R_Shank_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)

    pred_L_Trunk_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)
    pred_R_Trunk_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)

    ###################################################################################################################
    # Joint's Angles: L/R Elbow, L/R Shoulder, L/R Hip, and L/R Knee
    ###################################################################################################################

    pred_L_Elbow_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)
    pred_R_Elbow_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)

    pred_L_Shoulder_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                              null=True, blank=True)
    pred_R_Shoulder_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                              null=True, blank=True)

    pred_L_Hip_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                         null=True, blank=True)
    pred_R_Hip_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                         null=True, blank=True)

    pred_L_Knee_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                          null=True, blank=True)
    pred_R_Knee_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                          null=True, blank=True)

    ###################################################################################################################
    # Diver's Center of Mass (CoM)
    ###################################################################################################################

    diver_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    diver_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    ###################################################################################################################
    # Diver's Segment Center of Mass (CoM)
    # Segments: Head CoM, L/R Shank CoM, L/R Thigh CoM, L/R Up Arm CoM, L/R Forearm CoM, Mid Trunk CoM
    ###################################################################################################################

    Head_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    Head_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    L_Shank_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    L_Shank_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    R_Shank_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    R_Shank_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    L_Thigh_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    L_Thigh_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    R_Thigh_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    R_Thigh_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    L_Up_Arm_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    L_Up_Arm_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    R_Up_Arm_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    R_Up_Arm_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    L_Forearm_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    L_Forearm_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    R_Forearm_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    R_Forearm_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    Mid_Trunk_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    Mid_Trunk_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    ###################################################################################################################
    # CoM Centred Velocity and CoM Centred Acceleration
    # CoM Horizontal Velocity and CoM Horizontal Acceleration
    # CoM Vertical Velocity and CoM Vertical Acceleration
    ###################################################################################################################

    MOVEMENT_UNIT = (
        ('1', 'm/s'),
        ('2', 'm/s^2')
    )

    CoM_centred_velocity = models.FloatField(null=True, blank=True)
    CoM_centred_velocity_unit = models.CharField(null=True, blank=True, max_length=1, choices=MOVEMENT_UNIT)

    CoM_centred_acceleration = models.FloatField(null=True, blank=True)
    CoM_centred_acceleration_unit = models.CharField(null=True, blank=True, max_length=1, choices=MOVEMENT_UNIT)

    CoM_horizontal_velocity = models.FloatField(null=True, blank=True)
    CoM_horizontal_velocity_unit = models.CharField(null=True, blank=True, max_length=1, choices=MOVEMENT_UNIT)

    CoM_horizontal_acceleration = models.FloatField(null=True, blank=True)
    CoM_horizontal_acceleration_unit = models.CharField(null=True, blank=True, max_length=1, choices=MOVEMENT_UNIT)

    CoM_vertical_velocity = models.FloatField(null=True, blank=True)
    CoM_vertical_velocity_unit = models.CharField(null=True, blank=True, max_length=1, choices=MOVEMENT_UNIT)

    CoM_vertical_acceleration = models.FloatField(null=True, blank=True)
    CoM_vertical_acceleration_unit = models.CharField(null=True, blank=True, max_length=1, choices=MOVEMENT_UNIT)

    class Meta:
        ordering = ['frame_name']
        verbose_name = "Original predicted diver's 2D pose"
        verbose_name_plural = "Original predicted diver's 2D poses"

    def __str__(self):
        return 'Original predicted 2D diver pose: ' \
               'Video={}, Frame Name={}, Frame ID={} => (' \
               'pred_x_Nose={}, pred_y_Nose={}, prob_Nose_detected={}, ' \
               'pred_x_LEye={}, pred_y_LEye={}, prob_LEye_detected={}, ' \
               'pred_x_REye={}, pred_y_REye={}, prob_REye_detected={}, ' \
               'pred_x_LEar={}, pred_y_LEar={}, prob_LEar_detected={}, ' \
               'pred_x_REar={}, pred_y_REar={}, prob_REar_detected={}, ' \
               'pred_x_LSho={}, pred_y_LSho={}, prob_LSho_detected={}, ' \
               'pred_x_RSho={}, pred_y_RSho={}, prob_RSho_detected={}, ' \
               'pred_x_LElb={}, pred_y_LElb={}, prob_LElb_detected={}, ' \
               'pred_x_RElb={}, pred_y_RElb={}, prob_RElb_detected={}, ' \
               'pred_x_LWri={}, pred_y_LWri={}, prob_LWri_detected={}, ' \
               'pred_x_RWri={}, pred_y_RWri={}, prob_RWri_detected={}, ' \
               'pred_x_LHip={}, pred_y_LHip={}, prob_LHip_detected={}, ' \
               'pred_x_RHip={}, pred_y_RHip={}, prob_RHip_detected={}, ' \
               'pred_x_LKne={}, pred_y_LKne={}, prob_LKne_detected={}, ' \
               'pred_x_RKne={}, pred_y_RKne={}, prob_RKne_detected={}, ' \
               'pred_x_LAnk={}, pred_y_LAnk={}, prob_LAnk_detected={}, ' \
               'pred_x_RAnk={}, pred_y_RAnk={}, prob_RAnk_detected={}, ' \
               'num_detected_key_points={}, ' \
               'pred_L_Upper_arm_angle={}, pred_R_Upper_arm_angle={}, ' \
               'pred_L_Forearm_angle={}, pred_R_Forearm_angle={}, ' \
               'pred_L_Thigh_angle={}, pred_R_Thigh_angle={}, ' \
               'pred_L_Shank_angle={}, pred_R_Shank_angle={}, ' \
               'pred_L_Trunk_angle={}, pred_R_Trunk_angle={}, ' \
               'pred_L_Elbow_angle={}, pred_R_Elbow_angle={}, ' \
               'pred_L_Shoulder_angle={}, pred_R_Shoulder_angle={}, ' \
               'pred_L_Hip_angle={}, pred_R_Hip_angle={}, ' \
               'pred_L_Knee_angle={}, pred_R_Knee_angle={}, ' \
               'diver_x_CoM={}, diver_y_CoM={}, ' \
               'Head_CoM={}, Head_CoM={}, ' \
               'L_Shank_x_CoM={}, L_Shank_y_CoM={}, ' \
               'R_Shank_x_CoM={}, R_Shank_y_CoM={}, ' \
               'L_Thigh_x_CoM={}, L_Thigh_y_CoM={}, ' \
               'R_Thigh_x_CoM={}, R_Thigh_y_CoM={}, ' \
               'L_Up_Arm_x_CoM={}, L_Up_Arm_y_CoM={}, ' \
               'R_Up_Arm_x_CoM={}, R_Up_Arm_y_CoM={}, ' \
               'L_Forearm_x_CoM={}, L_Forearm_y_CoM={}, ' \
               'R_Forearm_x_CoM={}, R_Forearm_y_CoM={}, ' \
               'Mid_Trunk_x_CoM={}, Mid_Trunk_y_CoM={}, ' \
               'CoM_centred_velocity={}, CoM_centred_velocity_unit={}, ' \
               'CoM_centred_acceleration={}, CoM_centred_acceleration_unit={}, ' \
               'CoM_horizontal_velocity={}, CoM_horizontal_velocity_unit={}, ' \
               'CoM_horizontal_acceleration={}, CoM_horizontal_acceleration_unit={}, ' \
               'CoM_vertical_velocity={}, CoM_vertical_velocity_unit={}, ' \
               'CoM_vertical_acceleration={}, CoM_vertical_acceleration_unit={})'\
            .format(self.video, self.frame_name, self.frame_id,
                    self.pred_x_Nose, self.pred_y_Nose, self.prob_Nose_detected,
                    self.pred_x_LEye, self.pred_y_LEye, self.prob_LEye_detected,
                    self.pred_x_REye, self.pred_y_REye, self.prob_REye_detected,
                    self.pred_x_LEar, self.pred_y_LEar, self.prob_LEar_detected,
                    self.pred_x_REar, self.pred_y_REar, self.prob_REar_detected,
                    self.pred_x_LSho, self.pred_y_LSho, self.prob_LSho_detected,
                    self.pred_x_RSho, self.pred_y_RSho, self.prob_RSho_detected,
                    self.pred_x_LElb, self.pred_y_LElb, self.prob_LElb_detected,
                    self.pred_x_RElb, self.pred_y_RElb, self.prob_RElb_detected,
                    self.pred_x_LWri, self.pred_y_LWri, self.prob_LWri_detected,
                    self.pred_x_RWri, self.pred_y_RWri, self.prob_RWri_detected,
                    self.pred_x_LHip, self.pred_y_LHip, self.prob_LHip_detected,
                    self.pred_x_RHip, self.pred_y_RHip, self.prob_RHip_detected,
                    self.pred_x_LKne, self.pred_y_LKne, self.prob_LKne_detected,
                    self.pred_x_RKne, self.pred_y_RKne, self.prob_RKne_detected,
                    self.pred_x_LAnk, self.pred_y_LAnk, self.prob_LAnk_detected,
                    self.pred_x_RAnk, self.pred_y_RAnk, self.prob_RAnk_detected,
                    self.num_detected_key_points,
                    self.pred_L_Upper_arm_angle, self.pred_R_Upper_arm_angle,
                    self.pred_L_Forearm_angle, self.pred_R_Forearm_angle,
                    self.pred_L_Thigh_angle, self.pred_R_Thigh_angle,
                    self.pred_L_Shank_angle, self.pred_R_Shank_angle,
                    self.pred_L_Trunk_angle, self.pred_R_Trunk_angle,
                    self.pred_L_Elbow_angle, self.pred_R_Elbow_angle,
                    self.pred_L_Shoulder_angle, self.pred_R_Shoulder_angle,
                    self.pred_L_Hip_angle, self.pred_R_Hip_angle,
                    self.pred_L_Knee_angle, self.pred_R_Knee_angle,
                    self.diver_x_CoM, self.diver_y_CoM,
                    self.Head_CoM, self.Head_CoM,
                    self.L_Shank_x_CoM, self.L_Shank_y_CoM,
                    self.R_Shank_x_CoM, self.R_Shank_y_CoM,
                    self.L_Thigh_x_CoM, self.L_Thigh_y_CoM,
                    self.R_Thigh_x_CoM, self.R_Thigh_y_CoM,
                    self.L_Up_Arm_x_CoM, self.L_Up_Arm_y_CoM,
                    self.R_Up_Arm_x_CoM, self.R_Up_Arm_y_CoM,
                    self.L_Forearm_x_CoM, self.L_Forearm_y_CoM,
                    self.R_Forearm_x_CoM, self.R_Forearm_y_CoM,
                    self.Mid_Trunk_x_CoM, self.Mid_Trunk_y_CoM,
                    self.CoM_centred_velocity, self.get_CoM_centred_velocity_unit_display(),
                    self.CoM_centred_acceleration, self.get_CoM_centred_acceleration_unit_display(),
                    self.CoM_horizontal_velocity, self.get_CoM_horizontal_velocity_unit_display(),
                    self.CoM_horizontal_acceleration, self.get_CoM_horizontal_acceleration_unit_display(),
                    self.CoM_vertical_velocity, self.get_CoM_vertical_velocity_unit_display(),
                    self.CoM_vertical_acceleration, self.get_CoM_vertical_acceleration_unit_display())


class EnhancedPredictedDiver2DPose(models.Model):
    frame_name = models.CharField(null=True, blank=True, max_length=255)
    frame_id = models.PositiveIntegerField(null=True, blank=True)
    video = models.ForeignKey(Video, null=True, blank=True, on_delete=models.CASCADE)

    ###################################################################################################################
    # Joints (X, Y) and Probability of Joint Detection
    ###################################################################################################################

    pred_x_Nose = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_Nose = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_Nose_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_Nose_detected_below_min_thresh = models.BooleanField(default=False)

    pred_x_LEye = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LEye = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LEye_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_LEye_detected_below_min_thresh = models.BooleanField(default=False)
    LEye_x_and_y_values_copied_from_REye = models.BooleanField(default=False)

    pred_x_REye = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_REye = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_REye_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_REye_detected_below_min_thresh = models.BooleanField(default=False)
    REye_x_and_y_values_copied_from_LEye = models.BooleanField(default=False)

    pred_x_LEar = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LEar = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LEar_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_LEar_detected_below_min_thresh = models.BooleanField(default=False)
    LEar_x_and_y_values_copied_from_REar = models.BooleanField(default=False)

    pred_x_REar = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_REar = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_REar_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_REar_detected_below_min_thresh = models.BooleanField(default=False)
    REar_x_and_y_values_copied_from_LEar = models.BooleanField(default=False)

    pred_x_LSho = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LSho = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LSho_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_LSho_detected_below_min_thresh = models.BooleanField(default=False)
    LSho_x_and_y_values_copied_from_RSho = models.BooleanField(default=False)

    pred_x_RSho = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RSho = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RSho_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_RSho_detected_below_min_thresh = models.BooleanField(default=False)
    RSho_x_and_y_values_copied_from_LSho = models.BooleanField(default=False)

    pred_x_LElb = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LElb = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LElb_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_LElb_detected_below_min_thresh = models.BooleanField(default=False)
    LElb_x_and_y_values_copied_from_RElb = models.BooleanField(default=False)

    pred_x_RElb = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RElb = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RElb_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_RElb_detected_below_min_thresh = models.BooleanField(default=False)
    RElb_x_and_y_values_copied_from_LElb = models.BooleanField(default=False)

    pred_x_LWri = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LWri = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LWri_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_LWri_detected_below_min_thresh = models.BooleanField(default=False)
    LWri_x_and_y_values_copied_from_RWri = models.BooleanField(default=False)

    pred_x_RWri = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RWri = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RWri_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_RWri_detected_below_min_thresh = models.BooleanField(default=False)
    RWri_x_and_y_values_copied_from_LWri = models.BooleanField(default=False)

    pred_x_LHip = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LHip = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LHip_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_LHip_detected_below_min_thresh = models.BooleanField(default=False)
    LHip_x_and_y_values_copied_from_RHip = models.BooleanField(default=False)

    pred_x_RHip = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RHip = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RHip_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_RHip_detected_below_min_thresh = models.BooleanField(default=False)
    RHip_x_and_y_values_copied_from_LHip = models.BooleanField(default=False)

    pred_x_LKne = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LKne = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LKne_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_LKne_detected_below_min_thresh = models.BooleanField(default=False)
    LKne_x_and_y_values_copied_from_RKne = models.BooleanField(default=False)

    pred_x_RKne = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RKne = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RKne_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_RKne_detected_below_min_thresh = models.BooleanField(default=False)
    RKne_x_and_y_values_copied_from_LKne = models.BooleanField(default=False)

    pred_x_LAnk = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_LAnk = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_LAnk_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_LAnk_detected_below_min_thresh = models.BooleanField(default=False)
    LAnk_x_and_y_values_copied_from_RAnk = models.BooleanField(default=False)

    pred_x_RAnk = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    pred_y_RAnk = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    prob_RAnk_detected = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
                                           null=True, blank=True)
    prob_RAnk_detected_below_min_thresh = models.BooleanField(default=False)
    RAnk_x_and_y_values_copied_from_LAnk = models.BooleanField(default=False)

    ###################################################################################################################
    # Number of Detected Key Points (Joints)
    ###################################################################################################################

    num_detected_key_points = models.PositiveIntegerField(null=True, blank=True)

    ###################################################################################################################
    # Segment's Angles: L/R Upper Arm, L/R Forearm, L/R Thigh, L/R Shank, and L/R Trunk
    ###################################################################################################################

    pred_L_Upper_arm_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                               null=True, blank=True)
    pred_R_Upper_arm_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                               null=True, blank=True)

    pred_L_Forearm_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                             null=True, blank=True)
    pred_R_Forearm_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                             null=True, blank=True)

    pred_L_Thigh_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)
    pred_R_Thigh_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)

    pred_L_Shank_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)
    pred_R_Shank_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)

    pred_L_Trunk_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)
    pred_R_Trunk_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)

    ###################################################################################################################
    # Joint's Angles: L/R Elbow, L/R Shoulder, L/R Hip, and L/R Knee
    ###################################################################################################################

    pred_L_Elbow_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)
    pred_R_Elbow_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                           null=True, blank=True)

    pred_L_Shoulder_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                              null=True, blank=True)
    pred_R_Shoulder_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                              null=True, blank=True)

    pred_L_Hip_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                         null=True, blank=True)
    pred_R_Hip_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                         null=True, blank=True)

    pred_L_Knee_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                          null=True, blank=True)
    pred_R_Knee_angle = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(360.0)],
                                          null=True, blank=True)

    ###################################################################################################################
    # Diver's Center of Mass (CoM)
    ###################################################################################################################

    diver_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    diver_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    ###################################################################################################################
    # Diver's Segment Center of Mass (CoM)
    # Segments: Head CoM, L/R Shank CoM, L/R Thigh CoM, L/R Up Arm CoM, L/R Forearm CoM, Mid Trunk CoM
    ###################################################################################################################

    Head_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    Head_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    L_Shank_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    L_Shank_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    R_Shank_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    R_Shank_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    L_Thigh_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    L_Thigh_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    R_Thigh_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    R_Thigh_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    L_Up_Arm_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    L_Up_Arm_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    R_Up_Arm_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    R_Up_Arm_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    L_Forearm_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    L_Forearm_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    R_Forearm_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    R_Forearm_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    Mid_Trunk_x_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)
    Mid_Trunk_y_CoM = models.FloatField(validators=[MinValueValidator(0.0)], null=True, blank=True)

    ###################################################################################################################
    # CoM Centred Velocity and CoM Centred Acceleration
    # CoM Horizontal Velocity and CoM Horizontal Acceleration
    # CoM Vertical Velocity and CoM Vertical Acceleration
    ###################################################################################################################

    CoM_centred_velocity = models.FloatField(null=True, blank=True)
    CoM_centred_velocity_unit = models.CharField(null=True, blank=True, max_length=1,
                                                 choices=OriginalPredictedDiver2DPose.MOVEMENT_UNIT)

    CoM_centred_acceleration = models.FloatField(null=True, blank=True)
    CoM_centred_acceleration_unit = models.CharField(null=True, blank=True, max_length=1,
                                                     choices=OriginalPredictedDiver2DPose.MOVEMENT_UNIT)

    CoM_horizontal_velocity = models.FloatField(null=True, blank=True)
    CoM_horizontal_velocity_unit = models.CharField(null=True, blank=True, max_length=1,
                                                    choices=OriginalPredictedDiver2DPose.MOVEMENT_UNIT)

    CoM_horizontal_acceleration = models.FloatField(null=True, blank=True)
    CoM_horizontal_acceleration_unit = models.CharField(null=True, blank=True, max_length=1,
                                                        choices=OriginalPredictedDiver2DPose.MOVEMENT_UNIT)

    CoM_vertical_velocity = models.FloatField(null=True, blank=True)
    CoM_vertical_velocity_unit = models.CharField(null=True, blank=True, max_length=1,
                                                  choices=OriginalPredictedDiver2DPose.MOVEMENT_UNIT)

    CoM_vertical_acceleration = models.FloatField(null=True, blank=True)
    CoM_vertical_acceleration_unit = models.CharField(null=True, blank=True, max_length=1,
                                                      choices=OriginalPredictedDiver2DPose.MOVEMENT_UNIT)

    class Meta:
        ordering = ['frame_name']
        verbose_name = "Enhanced predicted diver's 2D pose"
        verbose_name_plural = "Enhanced predicted diver's 2D poses"

    def __str__(self):
        return 'Enhanced predicted 2D diver pose: ' \
               'Video={}, Frame Name={}, Frame ID={} => (' \
               'pred_x_Nose={}, pred_y_Nose={}, prob_Nose_detected={}, ' \
               'pred_x_LEye={}, pred_y_LEye={}, prob_LEye_detected={}, ' \
               'pred_x_REye={}, pred_y_REye={}, prob_REye_detected={}, ' \
               'pred_x_LEar={}, pred_y_LEar={}, prob_LEar_detected={}, ' \
               'pred_x_REar={}, pred_y_REar={}, prob_REar_detected={}, ' \
               'pred_x_LSho={}, pred_y_LSho={}, prob_LSho_detected={}, ' \
               'pred_x_RSho={}, pred_y_RSho={}, prob_RSho_detected={}, ' \
               'pred_x_LElb={}, pred_y_LElb={}, prob_LElb_detected={}, ' \
               'pred_x_RElb={}, pred_y_RElb={}, prob_RElb_detected={}, ' \
               'pred_x_LWri={}, pred_y_LWri={}, prob_LWri_detected={}, ' \
               'pred_x_RWri={}, pred_y_RWri={}, prob_RWri_detected={}, ' \
               'pred_x_LHip={}, pred_y_LHip={}, prob_LHip_detected={}, ' \
               'pred_x_RHip={}, pred_y_RHip={}, prob_RHip_detected={}, ' \
               'pred_x_LKne={}, pred_y_LKne={}, prob_LKne_detected={}, ' \
               'pred_x_RKne={}, pred_y_RKne={}, prob_RKne_detected={}, ' \
               'pred_x_LAnk={}, pred_y_LAnk={}, prob_LAnk_detected={}, ' \
               'pred_x_RAnk={}, pred_y_RAnk={}, prob_RAnk_detected={}, ' \
               'num_detected_key_points={}, ' \
               'pred_L_Upper_arm_angle={}, pred_R_Upper_arm_angle={}, ' \
               'pred_L_Forearm_angle={}, pred_R_Forearm_angle={}, ' \
               'pred_L_Thigh_angle={}, pred_R_Thigh_angle={}, ' \
               'pred_L_Shank_angle={}, pred_R_Shank_angle={}, ' \
               'pred_L_Trunk_angle={}, pred_R_Trunk_angle={}, ' \
               'pred_L_Elbow_angle={}, pred_R_Elbow_angle={}, ' \
               'pred_L_Shoulder_angle={}, pred_R_Shoulder_angle={}, ' \
               'pred_L_Hip_angle={}, pred_R_Hip_angle={}, ' \
               'pred_L_Knee_angle={}, pred_R_Knee_angle={}, ' \
               'diver_x_CoM={}, diver_y_CoM={}, ' \
               'Head_CoM={}, Head_CoM={}, ' \
               'L_Shank_x_CoM={}, L_Shank_y_CoM={}, ' \
               'R_Shank_x_CoM={}, R_Shank_y_CoM={}, ' \
               'L_Thigh_x_CoM={}, L_Thigh_y_CoM={}, ' \
               'R_Thigh_x_CoM={}, R_Thigh_y_CoM={}, ' \
               'L_Up_Arm_x_CoM={}, L_Up_Arm_y_CoM={}, ' \
               'R_Up_Arm_x_CoM={}, R_Up_Arm_y_CoM={}, ' \
               'L_Forearm_x_CoM={}, L_Forearm_y_CoM={}, ' \
               'R_Forearm_x_CoM={}, R_Forearm_y_CoM={}, ' \
               'Mid_Trunk_x_CoM={}, Mid_Trunk_y_CoM={}, ' \
               'CoM_centred_velocity={}, CoM_centred_velocity_unit={}, ' \
               'CoM_centred_acceleration={}, CoM_centred_acceleration_unit={}, ' \
               'CoM_horizontal_velocity={}, CoM_horizontal_velocity_unit={}, ' \
               'CoM_horizontal_acceleration={}, CoM_horizontal_acceleration_unit={}, ' \
               'CoM_vertical_velocity={}, CoM_vertical_velocity_unit={}, ' \
               'CoM_vertical_acceleration={}, CoM_vertical_acceleration_unit={})' \
            .format(self.video, self.frame_name, self.frame_id,
                    self.pred_x_Nose, self.pred_y_Nose, self.prob_Nose_detected,
                    self.pred_x_LEye, self.pred_y_LEye, self.prob_LEye_detected,
                    self.pred_x_REye, self.pred_y_REye, self.prob_REye_detected,
                    self.pred_x_LEar, self.pred_y_LEar, self.prob_LEar_detected,
                    self.pred_x_REar, self.pred_y_REar, self.prob_REar_detected,
                    self.pred_x_LSho, self.pred_y_LSho, self.prob_LSho_detected,
                    self.pred_x_RSho, self.pred_y_RSho, self.prob_RSho_detected,
                    self.pred_x_LElb, self.pred_y_LElb, self.prob_LElb_detected,
                    self.pred_x_RElb, self.pred_y_RElb, self.prob_RElb_detected,
                    self.pred_x_LWri, self.pred_y_LWri, self.prob_LWri_detected,
                    self.pred_x_RWri, self.pred_y_RWri, self.prob_RWri_detected,
                    self.pred_x_LHip, self.pred_y_LHip, self.prob_LHip_detected,
                    self.pred_x_RHip, self.pred_y_RHip, self.prob_RHip_detected,
                    self.pred_x_LKne, self.pred_y_LKne, self.prob_LKne_detected,
                    self.pred_x_RKne, self.pred_y_RKne, self.prob_RKne_detected,
                    self.pred_x_LAnk, self.pred_y_LAnk, self.prob_LAnk_detected,
                    self.pred_x_RAnk, self.pred_y_RAnk, self.prob_RAnk_detected,
                    self.num_detected_key_points,
                    self.pred_L_Upper_arm_angle, self.pred_R_Upper_arm_angle,
                    self.pred_L_Forearm_angle, self.pred_R_Forearm_angle,
                    self.pred_L_Thigh_angle, self.pred_R_Thigh_angle,
                    self.pred_L_Shank_angle, self.pred_R_Shank_angle,
                    self.pred_L_Trunk_angle, self.pred_R_Trunk_angle,
                    self.pred_L_Elbow_angle, self.pred_R_Elbow_angle,
                    self.pred_L_Shoulder_angle, self.pred_R_Shoulder_angle,
                    self.pred_L_Hip_angle, self.pred_R_Hip_angle,
                    self.pred_L_Knee_angle, self.pred_R_Knee_angle,
                    self.diver_x_CoM, self.diver_y_CoM,
                    self.Head_CoM, self.Head_CoM,
                    self.L_Shank_x_CoM, self.L_Shank_y_CoM,
                    self.R_Shank_x_CoM, self.R_Shank_y_CoM,
                    self.L_Thigh_x_CoM, self.L_Thigh_y_CoM,
                    self.R_Thigh_x_CoM, self.R_Thigh_y_CoM,
                    self.L_Up_Arm_x_CoM, self.L_Up_Arm_y_CoM,
                    self.R_Up_Arm_x_CoM, self.R_Up_Arm_y_CoM,
                    self.L_Forearm_x_CoM, self.L_Forearm_y_CoM,
                    self.R_Forearm_x_CoM, self.R_Forearm_y_CoM,
                    self.Mid_Trunk_x_CoM, self.Mid_Trunk_y_CoM,
                    self.CoM_centred_velocity, self.get_CoM_centred_velocity_unit_display(),
                    self.CoM_centred_acceleration, self.get_CoM_centred_acceleration_unit_display(),
                    self.CoM_horizontal_velocity, self.get_CoM_horizontal_velocity_unit_display(),
                    self.CoM_horizontal_acceleration, self.get_CoM_horizontal_acceleration_unit_display(),
                    self.CoM_vertical_velocity, self.get_CoM_vertical_velocity_unit_display(),
                    self.CoM_vertical_acceleration, self.get_CoM_vertical_acceleration_unit_display())
