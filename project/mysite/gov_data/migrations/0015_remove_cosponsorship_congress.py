# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0014_cosponsorship_congress'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='cosponsorship',
            name='congress',
        ),
    ]
