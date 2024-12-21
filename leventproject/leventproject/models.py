from django.db import models

# Fotoğraf ve tahminleri saklayacak model
class TextModel(models.Model):
    # El işaretinin tahmin sonucu
    texts = models.TextField()
    # Görselin path bilgisi
    image_path = models.CharField(max_length=500)

    # Django'nun Meta sınıfı ile veritabanı tablosunun adını belirliyoruz
    class Meta:
        db_table = 'text_model'  # Veritabanında bu tabloyu kullanacak

    def __str__(self):
        return f"Prediction: {self.texts}, Image Path: {self.image_path}"
