# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0006_auto_20160212_1528'),
    ]

    operations = [
        migrations.AlterField(
            model_name='senator',
            name='first',
            field=models.CharField(max_length=200, null=True, blank=True),
        ),
    ]
