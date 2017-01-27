# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Bills',
            new_name='Bill',
        ),
    ]
