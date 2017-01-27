# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0015_remove_cosponsorship_congress'),
    ]

    operations = [
        migrations.AlterField(
            model_name='senator',
            name='end',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='senator',
            name='start',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
