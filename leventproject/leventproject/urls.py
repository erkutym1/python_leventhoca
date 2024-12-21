from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.camera_view, name='camera_view'),
    path('process/', views.process_image, name='process_image'),
    path('upload_image_process/', views.upload_image_process, name='upload_image_process'),
    path('camera_page/', views.camera_page, name='camera_page'),
    path('foto_page/', views.foto_page, name='foto_page'),
    path('save-data/', views.save_data, name='save_data'),
    path('fetch-data/', views.fetch_data, name='fetch_data'),
    path('fetch_previous_images/', views.fetch_previous_images, name='fetch_previous_images'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)