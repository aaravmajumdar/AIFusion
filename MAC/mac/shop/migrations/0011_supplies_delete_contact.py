# Generated by Django 5.0.1 on 2024-02-17 14:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0010_alter_orders_amount'),
    ]

    operations = [
        migrations.CreateModel(
            name='Supplies',
            fields=[
                ('msg_id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
                ('email', models.CharField(default='', max_length=70)),
                ('phone', models.CharField(default='', max_length=70)),
                ('desc', models.CharField(default='', max_length=500)),
                ('filename', models.FileField(upload_to='images/')),
            ],
        ),
        migrations.DeleteModel(
            name='Contact',
        ),
    ]
