#This file is not actually called upon by Django, it uses the models.py file in "gov_data". 
#This was the original auto-generated file from our database

from __future__ import unicode_literals

from django.db import models


class Bills(models.Model):
    bill_id = models.IntegerField(primary_key=True, blank=True, null=False)
    title = models.TextField(blank=True, null=True)
    status = models.TextField(blank=True, null=True)
    introduced_date = models.TextField(blank=True, null=True)
    sponsor_id = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return self.title
    
    class Meta:
        db_table = 'bills'

class Cosponsorships(models.Model):
    action_id = models.IntegerField(primary_key=True, blank=True, null=False)
    bill = models.IntegerField(blank=True, null=True)
    cosponsor = models.IntegerField(blank=True, null=True)
    joined_date = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'cosponsorships'

class Senators(models.Model):
    id = models.IntegerField(primary_key=True, blank=True, null=False)  # AutoField?
    first = models.TextField(blank=True, null=True)
    last = models.TextField(blank=True, null=True)
    party = models.TextField(blank=True, null=True)
    class_field = models.TextField(db_column='class', blank=True, null=True)  # Field renamed because it was a Python reserved word.
    rank = models.TextField(blank=True, null=True)
    state = models.TextField(blank=True, null=True)
    congress = models.IntegerField(blank= True, null=True)

    class Meta:
        db_table = 'senators'
