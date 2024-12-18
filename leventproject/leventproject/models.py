# leventproject/models.py
from django.db import models

class TextModel(models.Model):
    texts = models.TextField()

    class Meta:
        db_table = 'text_model'  # Burada istediğiniz tablo adını belirleyebilirsiniz

    def __str__(self):
        return self.texts
