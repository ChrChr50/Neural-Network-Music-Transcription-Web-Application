from django.urls import path
from .views import FileHandler

urlpatterns = [
    path('', FileHandler, name = 'FileHandler')
]