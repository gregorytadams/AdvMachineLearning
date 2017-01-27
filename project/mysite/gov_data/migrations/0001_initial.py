# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Bills',
            fields=[
                ('bill_id', models.IntegerField(primary_key=True, serialize=False, blank=True)),
                ('title', models.TextField(null=True, blank=True)),
                ('status', models.TextField(null=True, blank=True)),
                ('introduced_date', models.TextField(null=True, blank=True)),
                ('sponsor_id', models.IntegerField(null=True, blank=True)),
            ],
            options={
                'db_table': 'bills',
            },
        ),
        migrations.CreateModel(
            name='Cosponsorships',
            fields=[
                ('action_id', models.IntegerField(primary_key=True, serialize=False, blank=True)),
                ('bill', models.IntegerField(null=True, blank=True)),
                ('cosponsor', models.IntegerField(null=True, blank=True)),
                ('joined_date', models.TextField(null=True, blank=True)),
            ],
            options={
                'db_table': 'cosponsorships',
            },
        ),
        migrations.CreateModel(
            name='Senators',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False, blank=True)),
                ('first', models.TextField(null=True, blank=True)),
                ('last', models.TextField(null=True, blank=True)),
                ('party', models.TextField(null=True, blank=True)),
                ('class_field', models.TextField(null=True, db_column='class', blank=True)),
                ('rank', models.TextField(null=True, blank=True)),
                ('state', models.TextField(null=True, blank=True)),
            ],
            options={
                'db_table': 'senators',
            },
        ),
    ]
