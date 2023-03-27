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
    path("check/", views.classify_gaze_2_label_in_pic),
    path("tmp_pic/", views.generate_tmp_pic),
    path("dataset/", views.get_dataset),
    path("interval/", views.get_interval_dataset),
    path("fix_word_map/", views.add_fixation_to_word),
    path("dataset/all/", views.get_all_time_dataset),
    path("nlp/", views.get_nlp_sequence),
    path("dataset/time/", views.get_timestamp_dataset),
    path("label_num/", views.get_label_num),
    path("test/",views.get_sentence_interval),

    path("test1/",views.get_handcraft_feature)

]
