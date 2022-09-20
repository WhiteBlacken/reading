from django.db import models


# Create your models here.


class Text(models.Model):
    content = models.CharField(max_length=1600)

    class Meta:
        db_table = "text"


class Dictionary(models.Model):
    en = models.CharField(max_length=100)
    zh = models.CharField(max_length=800)

    class Meta:
        db_table = "dictionary"
