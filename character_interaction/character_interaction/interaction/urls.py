from django.urls import path
from .views import process_text

urlpatterns = [
    path('process_text/', process_text, name='process_text'),
]
