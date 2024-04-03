from PIL import Image
from django.contrib.auth.models import User
from django.db import models


class Profile(models.Model):
    user = models.OneToOneField(User, null=True, on_delete=models.CASCADE)
    profile_image = models.ImageField(default='no_profile_image.png', null=True, blank=True, upload_to='profile_images')
    birth_date = models.DateField(null=True, blank=True)
    GENDER_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other')
    )
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, null=True, blank=True)
    date_created = models.DateTimeField(auto_now_add=True, null=True)

    def save(self, *args, **kwargs):
        super().save()

        if self.profile_image:
            img = Image.open(self.profile_image)

            if img.height > 522 or img.width > 350:
                img_output_size = (522, 350)
                img.thumbnail(img_output_size)
                img.save(self.profile_image.path)

    def __str__(self):
        return '{}: {} {}'.format(self.user.username, self.user.first_name, self.user.last_name)
