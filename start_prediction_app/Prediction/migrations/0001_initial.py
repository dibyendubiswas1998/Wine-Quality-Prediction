# Generated by Django 3.2 on 2021-05-08 03:12

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataColumn',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('col1', models.IntegerField(max_length=10)),
                ('col2', models.IntegerField(max_length=10)),
                ('col3', models.IntegerField(max_length=10)),
                ('col4', models.IntegerField(max_length=10)),
                ('col5', models.IntegerField(max_length=10)),
                ('col6', models.IntegerField(max_length=10)),
            ],
        ),
    ]