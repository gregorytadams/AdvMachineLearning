# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0008_auto_20160212_1535'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cosponsorship',
            name='joined_date',
            field=models.CharField(null=True, max_length=50, blank=True),
        ),
    ]
