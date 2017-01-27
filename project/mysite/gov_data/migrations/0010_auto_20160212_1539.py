# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0009_auto_20160212_1536'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bill',
            name='introduced_date',
            field=models.CharField(max_length=200, blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='bill',
            name='status',
            field=models.CharField(max_length=200, blank=True, null=True),
        ),
    ]
