# recognition/models.py
from django.db import models
from pgvector.django import VectorField

class Faces(models.Model):
    id = models.AutoField(primary_key=True)
    embedding = VectorField(dimensions=512, null=True)
    image = models.ImageField(upload_to='faces/', null=True, blank=True)
    name = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name or "Unknown"
