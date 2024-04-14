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
    re_path(r"^$", views.go_login),
    path("go_login/", views.go_login),  # 进入登录页面
    path("login/", views.my_login, name="login"),  # 登录逻辑
    path("choose/", views.choose_text),  # 选择文章页面
    path("reading/", views.reading),  # 进入阅读页面
    path("para/", views.get_para),  # 加载文章及翻译
    path("collect_page_data/", views.collect_page_data),  # 收集该页数据
    path("label/", views.go_label_page),  # 进入打标签页面
    path("collect_labels/", views.collect_labels),  # 收集标签
    path("get_pred/", views.get_pred),  # 结果预测
    path("page_info/", views.page_info),  # 记录下该页的信息
    path("analysis/", include("analysis.urls")),  # 数据分析及生成相关操作
    path("count_data_num/", views.count_data_num),
    path("process/", include("process.urls")),  # 新的数据分析及生成相关操作
]
