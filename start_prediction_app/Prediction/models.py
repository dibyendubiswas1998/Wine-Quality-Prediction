from django.db import models


# Create your models here.
class DataColumn(models.Model):
    try:
        col1 = models.DecimalField(max_digits=10, decimal_places=3)
        col2 = models.DecimalField(max_digits=10, decimal_places=3)
        col3 = models.DecimalField(max_digits=10, decimal_places=3)
        col4 = models.DecimalField(max_digits=10, decimal_places=3)
        col5 = models.DecimalField(max_digits=10, decimal_places=3)
        col6 = models.DecimalField(max_digits=10, decimal_places=3)

    except Exception as e:
        print(e)


class Path(models.Model):
    try:
        path = models.CharField(max_length=1000)

    except Exception as e:
        print(e)
