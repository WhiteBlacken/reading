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
from django.urls import include, path, re_path

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
    path("heatmap/all/", views.get_all_heatmap),
    path("heatmap/visual/", views.get_visual_heatmap),
    path("article/", views.article_2_csv),
    path("ccn_data/", views.get_cnn_dataset),
    path("row_level_fix/", views.get_row_level_fixations_map),
    path("row_level_heat/", views.get_row_level_heatmap),
    path("choose/", views.choose_text),
    path("fix_map_by_time/", views.get_fixation_by_time),
    path("speed/", views.get_speed),
    path("feature/", include("feature.urls")),
    path('topic_value/',views.get_topic_relevant),
    path('diff/',views.get_diff),
    path('att/',views.get_att),
]
