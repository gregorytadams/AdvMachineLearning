from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from .models import Bill, Senator, Cosponsorship, Leadership
from django.template import loader

#Created from scratch with aid from the Django documentation


def index(request):
    '''
    Renders the Home Page Template
    '''
    return render(request, 'gov_data/index.html')


def bills_index(request, congress = 114, bill_range = '1'):
    '''
    Returns a list of bills by Congress and pagination for the Bill Index Template
    '''
    bill_range = int(bill_range)
    end = 25 * bill_range
    start = end - 25
    bill_list = Bill.objects.filter(congress = congress).order_by('title')[start:end]
    next = str(bill_range + 1)
    if bill_range != 1:
        last = str(bill_range - 1)
    else:
        last = '1'
    context = {'bill_list': bill_list, 'start':start, 'end':end, 'next': next, 'last': last, 'congress': congress}
    return render(request, 'gov_data/bills_index.html', context)


def bill_detail(request, bill_id):
    '''
    Returns information about an individual bill for the Detail Template
    '''
    bill = Bill.objects.get(pk = bill_id)
    sponsor = Senator.objects.get(pk = str(bill.sponsor_id))
    cosponsor_actions = Cosponsorship.objects.filter(bill = bill_id)
    cosponsor_list = []
    for cosponsorship in cosponsor_actions:
        try:
            senator = Senator.objects.get(pk = str(cosponsorship.cosponsor))
            cosponsor_list.append(senator)
        except:
            pass
    context = {'bill': bill, 'sponsor': sponsor, 'cosponsor_list': cosponsor_list}
    return render(request, 'gov_data/bill_detail.html', context)


def senators_index(request, congress = 114):
    '''
    Returns a list of Senators by Congress to the Senator Index page
    '''
    senator_list = Senator.objects.filter(start__lte = congress, end__gte = congress).order_by('last')
    template = loader.get_template('gov_data/senators_index.html')
    context = {'senator_list': senator_list, }
    return HttpResponse(template.render(context, request))


def senator_detail(request, senator_id):
    '''
    Returns information about each Senator to the detail page, including our leadership score and all bill information
    '''
    senator = Senator.objects.get(pk = senator_id)
    bill_list = Bill.objects.filter(sponsor = senator_id)
    leadership_scores = Leadership.objects.filter(senator_id = senator_id).order_by('congress')
    score_list = []
    congress_list = []
    for congress in leadership_scores:
        score_list.append(congress.bill_success_score)
        congress_list.append(congress.congress)
    context = {'senator': senator, 'bill_list': bill_list, 'leadership_scores': leadership_scores, 'score_list': score_list, 'congress_list': congress_list}
    return render(request, 'gov_data/senator_detail.html', context)