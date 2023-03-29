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
    # 最重要的两个接口，画图+生成数据集，不过依赖于很多tools
    # 画图：按照时间画图
    path("all_time_pic/",views.get_all_time_pic),
    path("part_time_pic/",views.get_part_time_pic),
    path("dataset/",views.dataset_of_timestamp),
    path("dataset_all_time/",views.dataset_of_all_time),
]
