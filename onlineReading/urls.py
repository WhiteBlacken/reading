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

from django.urls import include, path, re_path

from . import views

urlpatterns = [
# <<<<<<< HEAD
#     re_path(r"^$", views.login_page),
#     path("admin/", admin.site.urls),
#     path("login/", views.login, name="login"),
#     path("text/", views.get_all_text_available),
#     path("para/", views.get_paragraph_and_translation),
#     path("cal/", views.cal),
#     path("reading/", views.reading),
#     path("label/", views.label),
#     path("data/", views.get_page_data),
#     path("label/send/", views.get_labels),
#     path("test/", views.Test),
#     path("heatmap/all/", views.get_all_heatmap),
#     path("heatmap/visual/", views.get_visual_heatmap),
#     path("article/", views.article_2_csv),
#     path("ccn_data/", views.get_cnn_dataset),
#     path("row_level_fix/", views.get_row_level_fixations_map),
#     path("row_level_heat/", views.get_row_level_heatmap),
#     path("choose/", views.choose_text),
#     path("fix_map_by_time/", views.get_fixation_by_time),
#     path("speed/", views.get_speed),
#     path("feature/", include("feature.urls")),
#     path('topic_value/', views.get_topic_relevant),
#     path('diff/', views.get_diff),
#     path('att/', views.get_att),
#     path('article/check/', views.check_article),
#     path('page_info/', views.get_page_info),
#     path('gaze/', views.get_pred),
#     # path('gaze_by_glass/', views.get_pred_by_glass),
#     path('semantic_attention_map/', views.get_semantic_attention_map),
#     # path('question_dataset/', views.get_question_dataset),
#     path('marker/',views.go_marker)
# =======
    re_path(r"^$", views.go_login),
    path("go_login/",views.go_login), # 进入登录页面
    path("login/", views.login, name="login"), # 登录逻辑
    path("choose/",views.choose_text), # 选择文章页面
    path("reading/", views.reading), # 进入阅读页面
    path("para/", views.get_para), # 加载文章及翻译
    path("collect_page_data/", views.collect_page_data), # 收集该页数据
    path("label/", views.go_label_page), # 进入打标签页面
    path("collect_labels/", views.collect_labels), # 收集标签

    path("analysis/", include("analysis.urls")), # 数据分析及生成相关操作

]
