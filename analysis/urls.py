"""onlineReading URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from . import views

urlpatterns = [
    path("all_time_pic/",views.get_all_time_pic),
    path("part_time_pic/",views.get_part_time_pic),
    path("dataset/",views.dataset_of_timestamp),
    path("dataset_all_time/",views.dataset_of_all_time),
    path("label_count/",views.count_label),
    path("word_index/", views.get_word_index),
    path("sent_domain/",views.sent_domain),
    path("dataset_new/", views.dataset_new),
    path("dataset_skip/", views.dataset_of_all_time_for_skip),
    path("get_pic_by_fix/", views.get_pic_by_fix)
]
