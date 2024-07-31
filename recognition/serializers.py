# recognition/serializers.py
from rest_framework import serializers
from .models import Faces

class FaceEmbeddingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Faces
        fields = "__all__"
        read_only_fields = ['embedding']
