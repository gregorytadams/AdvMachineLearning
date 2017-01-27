# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0007_auto_20160212_1533'),
    ]

    operations = [
        migrations.AlterField(
            model_name='senator',
            name='class_field',
            field=models.CharField(null=True, blank=True, max_length=200, db_column='class'),
        ),
        migrations.AlterField(
            model_name='senator',
            name='last',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='senator',
            name='party',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='senator',
            name='rank',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='senator',
            name='state',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]
