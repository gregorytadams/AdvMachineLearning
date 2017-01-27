# Create JSON Graph
#
# This creates a json file that tells us what the nodes in our sponsorship web 
# will be and what each edge will be

import networkx as nx
from networkx.readwrite import json_graph
from helper_functions import open_db, commit_db
import json

def fill_rep_dict(c, congress):
    '''
    This pulls data from our SQL database and adds it to a python dictionary, 
    which we will later use for graphing
    '''
    rep_dict = {}
    get_rep_codes(c, rep_dict, congress)
    get_rep_bills(c, rep_dict, congress)
    get_rep_relationships(rep_dict)
    get_leadership_scores(c, rep_dict, congress)
    return rep_dict    

def get_rep_codes(c, rep_dict, congress):
    '''
    This executes a SQL query to get the ID, names, and parties of each senator
    and adds these fields to our representative dictionary
    '''
    r = c.execute('SELECT id, first, last, party FROM SENATORS WHERE start <=' + congress + ' AND end >=' + congress)
    senators = r.fetchall()
    for rep in senators:
        s_id = rep[0]
        first = rep[1]
        last = rep[2]
        party = rep[3]
        name = first + ' ' + last
        rep_dict[s_id] = {}
        rep_dict[s_id]['bills'] = []
        rep_dict[s_id]['name'] = name
        rep_dict[s_id]['party'] = party

def get_rep_bills(c, rep_dict, congress):
    '''
    This executes a SQL query to find every bill that each senator has sponsored
    and adds it to rep_dict
    '''
    for s_id in rep_dict:
        r = c.execute('SELECT bill_id FROM bills WHERE sponsor_id = ' + str(s_id) + ' AND congress = ' + congress)
        bills = r.fetchall()
        for bill in bills:
            bill_id = bill[0]
            rep_dict[s_id]['bills'].append(bill_id)

def get_rep_relationships(rep_dict):
    '''
    This sums up the number of unique senators whose bills a given senator has
    co-sponsored (we call these relationships) and the number of unique 
    co-sponsors a given senator has signing on to his bills (total_cosponsors)
    '''
    for s_id in rep_dict:
        rep_dict[s_id]['relationships'] = {}
        total = 0 
        for bill_id in rep_dict[s_id]['bills']:
            r = c.execute('SELECT cosponsor_id FROM cosponsorships WHERE bill = ' + str(bill_id))
            cosponsors = r.fetchall()
            for rep in cosponsors:
                rep_id = rep[0]
                if rep_id not in rep_dict[s_id]['relationships']:
                    rep_dict[s_id]['relationships'][rep_id] = 0
                rep_dict[s_id]['relationships'][rep_id] += 1

def get_leadership_scores(c, rep_dict, congress):
    for s_id in rep_dict:
        r = c.execute('SELECT bill_success_score FROM leadership WHERE senator_id = ' + str(s_id) + ' AND congress = ' + congress)
        score = r.fetchall()
        try: 
            score = score[0][0]
        except:
            score = 0
        rep_dict[s_id]['leadership_score'] = score

def graph_nodes(rep_dict):
    '''
    This generates the relevant data for the nodes in our networkx graph,
    where the label is the name of the Senator, the colour corresponds to their
    party, and the size corresponds to total_cosponsors
    '''
    graph = nx.Graph()

    for rep in rep_dict:
        name = rep_dict[rep]['name']
        leadership_score = rep_dict[rep]['leadership_score']
        if rep_dict[rep]['party'] == 'Republican':
            graph.add_nodes_from([rep], party = 'r', label = name, size = leadership_score)
        elif rep_dict[rep]['party'] == 'Democrat':
            graph.add_nodes_from([rep], party = 'd', label = name, size = leadership_score)    
        else: 
            graph.add_nodes_from([rep], party = 'o', label = name, size = leadership_score)        
    pos = nx.random_layout(graph)
    
    return graph,pos

def graph_edges(rep_dict, graph, pos):
    '''
    Using the networkx graph object initialized in graph_nodes, this adds edges
    between the nodes, where each edge represents a relationship between the two
    senators.
    '''
    checked_list = []
    all_actions = []
    for rep in graph.nodes():
        for cosponsor in rep_dict[rep]['relationships']:
                if (rep, cosponsor) not in checked_list and (cosponsor, rep) not in checked_list:
                    num_sponsorships = rep_dict[rep]['relationships'][cosponsor]
                    if cosponsor in rep_dict:
                        if rep in rep_dict[cosponsor]['relationships']:
                            num_sponsorships += rep_dict[cosponsor]['relationships'][rep]
                    checked_list.append((rep, cosponsor))
                    all_actions.append(num_sponsorships)
                    if num_sponsorships >= 10:
                        graph.add_edge(cosponsor, rep, weight = num_sponsorships)

def write_json(graph, congress):
    '''
    This dumps the graph data from graph_nodes and graph_edges into a JSON file
    '''
    d = json_graph.node_link_data(graph)
    json.dump(d, open('gov_data/static/gov_data/force_json/'+ str(congress) + '_force.json','w'))

if __name__ == '__main__':
    c, db = open_db('GovData')
    start = 108
    for congress in range(start, 110):
        print("congress {} running".format(congress))
        congress = str(congress)
        rep_dict = fill_rep_dict(c, congress)
        graph,pos = graph_nodes(rep_dict)
        graph_edges(rep_dict, graph, pos)   
        write_json(graph, congress)
        print("congress {} done".format(congress))
    commit_db(db)
