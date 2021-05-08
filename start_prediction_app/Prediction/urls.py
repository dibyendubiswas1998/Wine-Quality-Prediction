from django.urls import path
from .import views

urlpatterns = [
    path('', views.Predict_White_Wine, name="Predict_White_Wine"),
    path('bad/', views.Result_Bad, name="Result_Bad"),
    path('good/', views.Result_Good, name="Result_Good"),

]
