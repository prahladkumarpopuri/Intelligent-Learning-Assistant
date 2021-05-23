from django.db import models

# Create your models here.
class QP(models.Model):
    paragraph = models.TextField()
    question = models.CharField(max_length=250)