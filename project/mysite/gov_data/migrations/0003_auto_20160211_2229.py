# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0002_auto_20160211_2223'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Cosponsorships',
            new_name='Cosponsorship',
        ),
        migrations.RenameModel(
            old_name='Senators',
            new_name='Senator',
        ),
    ]
