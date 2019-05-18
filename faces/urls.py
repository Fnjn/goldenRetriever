from django.conf.urls import url

from . import views

app_name = 'faces'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^upload_success/$', views.uploadSuccess, name='uploadSuccess'),
    url(r'^insertsuccess/$', views.insertSuccess, name='insertSuccess'),
    url(r'^displayimg/$', views.DispalyImgView.as_view(), name='displayimg'),
    url(r'^compareupload/$', views.compareUploadView.as_view(), name='compareUpload'),
    url(r'^searchupload/$', views.searchUploadView.as_view(), name='searchUpload'),
    url(r'^insertupload/$', views.insertUploadView.as_view(), name='insertUpload'),
    url(r'^compare/$', views.compare, name='compare'),
    url(r'^search/$', views.search, name='search'),
    url(r'^train/$', views.train, name='train'),
]
