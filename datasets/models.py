from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from camera.models import CameraCalibration
from users.models import Profile


class Video(models.Model):
    VIDEO_ROTATION = (
        ('1', '90 degrees clockwise'),
        ('2', '180 degrees clockwise'),
        ('3', '270 degrees clockwise'),
    )

    VIDEO_DIVING_ANALYSIS_STATUS = (
        ('N', _('Not Started')),
        ('O', _('Ongoing')),
        ('C', _('Completed')),
    )

    VIDEO_ANALYSIS_STATUS = (
        ("NStarted", _("Not Started")),
        ("Started", _("Started")),
        ("Task0001", _("Bounding boxes estimation (diver, springboard, splash)")),
        ("Task0002", _("Estimating springboard height")),
        ("Task0003", _("Estimating springboard angles")),
        ("Task0004", _("Estimating springboard velocity and acceleration")),
        ("Task0005", _("Estimating splash percentage")),
        ("Task0006", _("Estimating the diver's trajectory using the center of the bounding box")),
        ("Task0007", _("Diver's 2D pose estimation")),
        ("Task0008", _("Estimating the center of mass of the diver's 2d pose")),
        ("Task0009", _("Estimating the diver's trajectory using the center of mass of the diver's 2d pose")),
        ("Task0010", _("Estimating the diver's center of mass velocity and acceleration")),
        ("Task0011", _("Estimating angles in the diver's body (arms, legs, and trunk)")),
        ("Task0012", _("Estimating kinematic metric 1: maximum springboard depression")),
        ("Task0013", _("Estimating kinematic metric 2: water contact")),
        ("Task0014", _("Estimating kinematic metric 3: the highest point in the preparation phase")),
        ("Task0015", _("Estimating kinematic metric 4: touch down")),
        ("Task0016", _("Estimating kinematic metric 5: the highest point in the flight time")),
        ("Task0017", _("Estimating kinematic metric 6: lift-off")),
        ("Task0018", _("Estimating kinematic metric 7: opening phase")),
        ("Complete", _("Completed"))
    )

    PREDICTED_MOVING_SPRINGBOARD = (
        ('T', _('Top springboard')),
        ('B', _('Bottom springboard')),
        ('U', _('Unknown'))
    )

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    diving_type = models.ForeignKey(DivingType, null=True, blank=True, on_delete=models.DO_NOTHING)
    athlete = models.ForeignKey(Profile, null=True, blank=True, on_delete=models.DO_NOTHING)
    name = models.CharField(max_length=255, unique=True)
    extension = models.CharField(default='mp4', max_length=8)
    uploaded_video_file = models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_file = models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    num_frames = models.IntegerField(null=True, blank=True)
    duration_in_sec = models.FloatField(null=True, blank=True)
    fps = models.FloatField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    uploaded_video_rotation_option = models.CharField(null=True, blank=True, max_length=1, choices=VIDEO_ROTATION)
    video_diving_analysis_status = models.CharField(max_length=1, choices=VIDEO_DIVING_ANALYSIS_STATUS, default='N')
    video_analysis_status = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='NStarted')
    predicted_moving_springboard = models.CharField(max_length=1, choices=PREDICTED_MOVING_SPRINGBOARD, default='U')
    video_with_original_predicted_bboxes = models.FileField(upload_to='videos', null=True, blank=True,
                                                            max_length=512)
    video_with_predicted_bboxes_without_bbox_duplication = models.FileField(upload_to='videos', null=True, blank=True,
                                                                            max_length=512)
    video_with_enhanced_predicted_bboxes = models.FileField(upload_to='videos', null=True, blank=True,
                                                               max_length=512)
    video_with_wider_bboxes_for_diver_2D_pose_estimation = models.FileField(upload_to='videos', null=True, blank=True,
                                                               max_length=512)
    video_with_springboard_height = models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_springboard_angles = models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_splash_percentage = models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_the_center_of_the_diver_bbox_tracking_line = \
        models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_original_diver_2d_pose_estimation = \
        models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_enhanced_diver_2d_pose_estimation = \
        models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_the_original_diver_2D_pose_CoM = \
        models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_the_enhanced_diver_2D_pose_CoM = \
        models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_angles_in_the_original_diver_2D_pose = \
        models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    video_with_angles_in_the_enhanced_diver_2D_pose = \
        models.FileField(upload_to='videos', null=True, blank=True, max_length=512)
    camera_calibration = models.ForeignKey(CameraCalibration, null=True, blank=True, on_delete=models.DO_NOTHING)
    creation_date = models.DateTimeField(default=timezone.now, null=True, blank=True)

    task_0001_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0001')
    task_0001_start_date = models.DateTimeField(null=True, blank=True)
    task_0001_end_date = models.DateTimeField(null=True, blank=True)

    task_0002_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0002')
    task_0002_start_date = models.DateTimeField(null=True, blank=True)
    task_0002_end_date = models.DateTimeField(null=True, blank=True)

    task_0003_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0003')
    task_0003_start_date = models.DateTimeField(null=True, blank=True)
    task_0003_end_date = models.DateTimeField(null=True, blank=True)

    task_0004_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0004')
    task_0004_start_date = models.DateTimeField(null=True, blank=True)
    task_0004_end_date = models.DateTimeField(null=True, blank=True)

    task_0005_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0005')
    task_0005_start_date = models.DateTimeField(null=True, blank=True)
    task_0005_end_date = models.DateTimeField(null=True, blank=True)

    task_0006_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0006')
    task_0006_start_date = models.DateTimeField(null=True, blank=True)
    task_0006_end_date = models.DateTimeField(null=True, blank=True)

    task_0007_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0007')
    task_0007_start_date = models.DateTimeField(null=True, blank=True)
    task_0007_end_date = models.DateTimeField(null=True, blank=True)

    task_0008_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0008')
    task_0008_start_date = models.DateTimeField(null=True, blank=True)
    task_0008_end_date = models.DateTimeField(null=True, blank=True)

    task_0009_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0009')
    task_0009_start_date = models.DateTimeField(null=True, blank=True)
    task_0009_end_date = models.DateTimeField(null=True, blank=True)

    task_0010_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0010')
    task_0010_start_date = models.DateTimeField(null=True, blank=True)
    task_0010_end_date = models.DateTimeField(null=True, blank=True)

    task_0011_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0011')
    task_0011_start_date = models.DateTimeField(null=True, blank=True)
    task_0011_end_date = models.DateTimeField(null=True, blank=True)

    task_0012_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0012')
    task_0012_start_date = models.DateTimeField(null=True, blank=True)
    task_0012_end_date = models.DateTimeField(null=True, blank=True)

    task_0013_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0013')
    task_0013_start_date = models.DateTimeField(null=True, blank=True)
    task_0013_end_date = models.DateTimeField(null=True, blank=True)

    task_0014_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0014')
    task_0014_start_date = models.DateTimeField(null=True, blank=True)
    task_0014_end_date = models.DateTimeField(null=True, blank=True)

    task_0015_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0015')
    task_0015_start_date = models.DateTimeField(null=True, blank=True)
    task_0015_end_date = models.DateTimeField(null=True, blank=True)

    task_0016_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0016')
    task_0016_start_date = models.DateTimeField(null=True, blank=True)
    task_0016_end_date = models.DateTimeField(null=True, blank=True)

    task_0017_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0017')
    task_0017_start_date = models.DateTimeField(null=True, blank=True)
    task_0017_end_date = models.DateTimeField(null=True, blank=True)

    task_0018_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0018')
    task_0018_start_date = models.DateTimeField(null=True, blank=True)
    task_0018_end_date = models.DateTimeField(null=True, blank=True)

    task_0019_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0019')
    task_0019_start_date = models.DateTimeField(null=True, blank=True)
    task_0019_end_date = models.DateTimeField(null=True, blank=True)

    task_0020_name = models.CharField(max_length=8, choices=VIDEO_ANALYSIS_STATUS, default='Task0020')
    task_0020_start_date = models.DateTimeField(null=True, blank=True)
    task_0020_end_date = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['name']

    def get_absolute_url(self):
        return reverse('datasets:video-detail', kwargs={'pk': self.pk})

    def __str__(self):
        return self.name
