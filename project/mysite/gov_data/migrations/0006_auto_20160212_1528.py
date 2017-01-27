# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0005_auto_20160212_1521'),
    ]

    operations = [
        migrations.AlterField(
            model_name='senator',
            name='first',
            field=models.TextField(max_length=200, null=True, blank=True),
        ),
    ]
