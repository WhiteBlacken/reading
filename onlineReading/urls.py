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
from django.contrib import admin
from django.urls import path, re_path

import action.views
from . import views

urlpatterns = [
    re_path(r"^$", views.login_page),
    path("admin/", admin.site.urls),
    path("login/", views.login, name="login"),
    path("text/", views.get_all_text_available),
    path("para/", views.get_paragraph_and_translation),
    path("cal/", views.cal),
    path("reading/", views.reading),
    path("label/", views.label),
    path("data/", views.get_page_data),
    path("label/send/", views.get_labels),
    path("test/", views.test_dispersion),
    path("dispersion/", views.get_dispersion),
    path("cm/<int:k>/", views.cm_2_pixel_test),
    path("analysis/", views.analysis),
    path("analysis_1/", views.analysis_1),
    path("heatmap/visual/", views.get_visual_heatmap),
    path("heatmap/nlp/", views.get_nlp_heatmap),
    path("motion/", views.test_motion),
    path("save_gaze/", views.get_gaze_data_pic),
    path("heatmap/",views.get_heatmap),
    path("visual/attention/",views.visual_attention),
]
