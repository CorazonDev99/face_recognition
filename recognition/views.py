import numpy as np
import cv2
from rest_framework import generics, status
from rest_framework.response import Response
from insightface.app import FaceAnalysis
from .models import Faces
from .serializers import FaceEmbeddingSerializer
from pgvector.django import L2Distance
from numpy.linalg import norm
from numpy import dot

app = FaceAnalysis(name="buffalo_sc", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256, 256))

class FaceEmbeddingCreateView(generics.CreateAPIView):
    queryset = Faces.objects.all()
    serializer_class = FaceEmbeddingSerializer

    def perform_create(self, serializer):
        image = self.request.FILES['image']
        img_array = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        faces = app.get(img)
        if faces:
            embedding = faces[0].embedding
            serializer.save(embedding=embedding, image=image)
        else:
            raise ValueError("No face detected in the uploaded image.")

class FaceSearchView(generics.GenericAPIView):
    serializer_class = FaceEmbeddingSerializer

    def post(self, request, *args, **kwargs):
        image = request.FILES['image']
        img_array = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        faces = app.get(img)
        if faces:
            query_embedding = faces[0].embedding
            fbase = Faces.objects.alias(distance=L2Distance('embedding', query_embedding)).order_by('distance')
            if fbase:
                closest_face = fbase.first()
                similarity = dot(query_embedding, closest_face.embedding) / (
                            norm(query_embedding) * norm(closest_face.embedding))
                similarity_percentage = similarity * 100
                serializer = self.get_serializer(closest_face)
                response_data = serializer.data
                response_data['similarity_percentage'] = similarity_percentage
                return Response(response_data)
            return Response({"message": "No similar face found."}, status=status.HTTP_404_NOT_FOUND)
        else:
            return Response({"message": "No face detected in the uploaded image."}, status=status.HTTP_400_BAD_REQUEST)