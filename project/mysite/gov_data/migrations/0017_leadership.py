# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('gov_data', '0016_auto_20160303_1520'),
    ]

    operations = [
        migrations.CreateModel(
            name='Leadership',
            fields=[
                ('id', models.AutoField(serialize=False, primary_key=True, verbose_name='ID', auto_created=True)),
                ('senator_id', models.IntegerField(blank=True, null=True)),
                ('congress', models.IntegerField(blank=True, null=True)),
                ('bill_success_score', models.IntegerField(blank=True, null=True)),
                ('cosponsors_out', models.IntegerField(blank=True, null=True)),
                ('cosponsors_in', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'leadership',
            },
        ),
    ]
