# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0003_auto_20160211_2229'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bill',
            name='sponsor_id',
            field=models.ForeignKey(null=True, to='gov_data.Senator'),
        ),
        migrations.AlterField(
            model_name='cosponsorship',
            name='cosponsor',
            field=models.ForeignKey(null=True, to='gov_data.Senator'),
        ),
    ]
