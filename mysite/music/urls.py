from django.urls import path
from . import views

urlpatterns = [
    path('', views.FileHandler, name = 'FileHandler'),
    path('confirmation/', views.Confirmation, name = 'Confirmation')
]
