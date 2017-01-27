#Database Set Up Code
import sqlite3
from helper_functions import make_url_request, open_db, commit_db

def add_rep_to_db(c):
    '''
    Takes a senator and adds them to the database.
    '''
    url = "https://www.govtrack.us/api/v2/role?startdate__gt=1950-01-01&&role_type__exact=senator&&limit=6000"
    data = make_url_request(url)
    for senator in data['objects']:
        s_id = senator['person']['id']
        first = senator['person']['firstname']
        last = senator['person']['lastname']
        party = senator['party']
        s_class = senator['senator_class']
        state = senator['state']
        start = senator['congress_numbers'][0]
        end = senator['congress_numbers'][-1]

        try:
            db_args = (s_id, first, last, party, s_class, state, start, end)
            c.execute('INSERT INTO senators VALUES(?,?,?,?,?,?,?,?)', db_args)  
        except:
            r = c.execute('SELECT start, end FROM senators WHERE id =' + str(s_id))
            date_list = r.fetchall()
            current_start = date_list[0][0]
            current_end = date_list[0][1]
            if start < int(current_start):
                c.execute('UPDATE senators SET start =' + str(start) + ' WHERE id =' + str(s_id))
            elif end > int(current_end):
                c.execute('UPDATE senators SET end =' + str(end) + ' WHERE id =' + str(s_id))

def add_sponsored_bills(c):
    '''
    Adds senators' sponsored bills to the database
    '''
    r = c.execute('SELECT id FROM senators')
    senators = r.fetchall()
    for s_id in senators:
        s_id = s_id[0]
        url = "https://www.govtrack.us/api/v2/bill?limit=6000&&sponsor__exact=" + str(s_id)
        data = make_url_request(url)
        for bill in data['objects']:
            bill_id = bill['id']
            title = bill['title']
            status = bill['current_status']
            introduced_date = bill['introduced_date']
            sponsor_id = bill['sponsor']['id']
            congress = bill['congress']
            db_args = (bill_id, title, status, introduced_date, sponsor_id, congress)
            c.execute('INSERT INTO bills VALUES(?,?,?,?,?,?)', db_args)
        
def add_cosponsorships(c):
    '''
    similar to above, adds data about cosponsorships to the database.
    '''
    r = c.execute('SELECT bill_id FROM bills')
    bills = r.fetchall()
    for bill in bills:
        try:
            bill = bill[0]
            url = 'https://www.govtrack.us/api/v2/cosponsorship?bill__exact=' + str(bill)
            data = make_url_request(url)
            for person in data['objects']:
                action_id = person['id']
                cosponsor = person['person']
                joined_date = person['joined']
                db_args = (action_id, bill, cosponsor, joined_date)
                c.execute('INSERT INTO cosponsorships VALUES(?,?,?,?)', db_args)
        except:
            pass

if __name__ == '__main__':
    c, db = open_db('GovData')
    add_rep_to_db(c)
    add_sponsored_bills(c)
    add_cosponsorships(c)
    commit_db(db)