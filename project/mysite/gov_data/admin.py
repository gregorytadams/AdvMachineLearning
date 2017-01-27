from django.contrib import admin
from .models import Bill, Cosponsorship, Senator
#Follows the Django Tutorial Online, does not really pertain to our project.

class CosponsorhsipInLine(admin.TabularInline):
    model = Cosponsorship
    extra = 0

class BillInLine(admin.TabularInline):
    model = Bill
    extra = 0

class SenatorAdmin(admin.ModelAdmin):
    inlines = [BillInLine, CosponsorhsipInLine]
    list_display = ('last', 'state', 'party')
    list_filter = ['party', 'state', 'last']
    search_fields = ['last']

class BillAdmin(admin.ModelAdmin):
    list_display = ('title', 'status', 'introduced_date')

admin.site.register(Bill, BillAdmin)

admin.site.register(Senator, SenatorAdmin)

