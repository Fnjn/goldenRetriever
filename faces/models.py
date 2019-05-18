from django.db import models
from django.utils import timezone
from base64 import b64encode

def generate_filename(self, filename):
    url = "UploadedFiles/%s_%s" % (timezone.now().strftime('%Y-%m-%d-%H-%M-%S'), b64encode(filename.encode()))
    return url

class UploadFile(models.Model):
    file_field = models.FileField(upload_to=generate_filename)

    def __str__(self):
        return self.file_field.url
