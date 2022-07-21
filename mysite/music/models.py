from django.db import models

class Audio(models.Model):
    docfile = models.FileField(upload_to = 'documents/%Y/%m/%d')