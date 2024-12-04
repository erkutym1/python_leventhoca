from django.urls import path
from . import views

urlpatterns = [
    path('', views.camera_view, name='camera_view'),
    path('predict/', views.predict_view(), name='predict_view'),

]
