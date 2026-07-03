from django.contrib import admin
from django.urls import include, path
from django.contrib.auth import views as auth_views
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),

    # Redirect homepage to login
    path("", RedirectView.as_view(url="/login/", permanent=False), name="home"),

    # Authentication
    path("login/", auth_views.LoginView.as_view(template_name="login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),

    # Apps
    path("orders/", include("orders.urls")),
    path("support/", include("support.urls")),
]