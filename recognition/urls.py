from django.urls import path
from .views import FaceEmbeddingCreateView, FaceSearchView

urlpatterns = [
    path('upload/', FaceEmbeddingCreateView.as_view(), name='upload-face'),
    path('search/', FaceSearchView.as_view(), name='search-face'),
]
