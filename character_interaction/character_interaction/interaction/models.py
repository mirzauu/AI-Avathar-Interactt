from django.db import models


class UserInput(models.Model):
    text = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
