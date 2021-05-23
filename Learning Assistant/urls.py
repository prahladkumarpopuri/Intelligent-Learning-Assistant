from django.contrib import admin
from . import views
from django.urls import path, include
from django.conf.urls import url

urlpatterns=[
    path('',views.basic, name='basic_view'),
]