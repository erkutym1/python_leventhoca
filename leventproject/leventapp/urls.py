from django.urls import path
from . import views

urlpatterns = [
    path('', views.camera_view, name='camera_view'),
    path('process/', views.process_image, name='process_image'),

]
