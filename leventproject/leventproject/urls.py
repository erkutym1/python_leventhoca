from django.urls import path
from . import views

urlpatterns = [
    path('', views.camera_view, name='camera_view'),
    path('process/', views.process_image, name='process_image'),
    path('oyun_sayfasi/', views.oyun_sayfasi, name='oyun_sayfasi'),
    path('camera_page/', views.camera_page, name='camera_page'),
    path('foto_page/', views.foto_page, name='foto_page'),
    path('save-data/', views.save_data, name='save_data'),
    path('fetch-data/', views.fetch_data, name='fetch_data'),
]
