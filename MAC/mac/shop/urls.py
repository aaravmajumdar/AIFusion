from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="ShopHome"),
    path("contact/", views.supplies, name="ContactUs"),
    path("tracker/", views.tracker, name="TrackingStatus"),
]
