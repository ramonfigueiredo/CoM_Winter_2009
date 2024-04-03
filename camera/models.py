from django.db import models
from django.urls import reverse
from django.utils import timezone


class CameraCalibration(models.Model):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(null=True, blank=True)
    pixel_to_cm = models.FloatField()
    creation_date = models.DateTimeField(default=timezone.now, null=True, blank=True)

    class Meta:
        ordering = ['name']

    def get_absolute_url(self):
        return reverse('camera:camera-calibration-detail', kwargs={'pk': self.pk})

    def __str__(self):
        return self.name
