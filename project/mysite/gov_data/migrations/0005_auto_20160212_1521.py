# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0004_auto_20160212_1519'),
    ]

    operations = [
        migrations.RenameField(
            model_name='bill',
            old_name='sponsor_id',
            new_name='sponsor',
        ),
    ]
