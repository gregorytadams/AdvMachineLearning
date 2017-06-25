# network.py

import networkx as nx
from random import choice, randint
from numpy.random import choice as npchoice
from itertools import combinations
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
print("Imports Finished!")

def make_dummy_data(num_bills = 400, num_senators = 10, max_cosponsors = 2):
    '''
    The web scraper is slow, so I initially built my networks with dummy data before using the real data.
    This is also useful for testing on smaller datasets, so I thought I'd leave it here.
    '''
    bills = ["S.{}".format(i+1) for i in range(num_bills)]
    senators = ["Senator{}[{}-NA]".format(i, choice(["R","D"])) for i in range(num_senators)]
    weights = [randint(1,100) for i in range(num_senators)]
    s = sum(weights)
    p_dist = map(lambda x: x/s, weights) # needs to sum to one (or at least get within floating point error)
    master_dict = {}
    for key in bills:
        master_dict[key] = [choice(senators), list(npchoice(senators, randint(0,max_cosponsors), p_dist))]
        if master_dict[key][0] in master_dict[key][1]: 
            master_dict[key][1].remove(master_dict[key][0])
    return master_dict


def open_convert_json(filename):
    '''
    Opens the json files from the web scraper and converts their keys to ints 

    input: 
    filename, a path to the json from the current directory 

    output:
    ne_dict, the dictionary represented by the json with ints as keys
    '''
    with open(filename) as f:
        old_dict = json.loads(f.read())
    new_dict = {}
    for key in old_dict: # I should have used pickle, but this doesn't take enough time to justify changing it. Lesson learned.
        new_dict[int(key)] = old_dict[key]
    return new_dict


def construct_people_network(master_dict, undirected):
    '''
    People are nodes, edges are cosponsorships
    For directed graph, if person a cosponsored person b's bill, theres an edge from a to b

    inputs: 
    master_dict, a dictionary with bill number as keys and a list/tuple len 2 [sponsor, [cosponsor1, cosponsor2, ...]]
    undirected, bool True for normal graph, False for directed graph 
    
    output:
    graph, a networkx graph object
    '''
    if undirected:
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()
    # Creates network.  This is a lot quicker than calling update_network (it doesn't have to check nodes every time)
    edges = {}
    for key in master_dict:
        for person in master_dict[key][1]:
            single_edge = (person, master_dict[key][0])
            if single_edge in edges:
                edges[single_edge] += 1
            else:
                edges[single_edge] = 1
    # make the graphs, weighted by occurrence
    for combo in edges:
        graph.add_edge(combo[0], combo[1], weight=edges[combo])
    return graph


def visualize_graph(graph, weight_cutoff = 5, title = "my_network", node_colors = 'red', save= False, \
                    show = False, include_labels = False):
    '''
    Builds a visualization of a networkx graph.  Mildly hardcoded for people network at the moment.

    Inputs:
    graph, a networkx graph object
    weight_cutoff, a cutoff for edges that get visualized.  too high makes a sparse network; too low makes a dense one 
    title, the title of the saved visualization
    node_colors, default node color.  People network turns democrats blue.
    save, bool -- save the png?
    show, bool -- show the png?
    include_labels, bool -- include the names of each senator?
    '''
    my_plot = nx.spring_layout(graph, k = 0.1, iterations = 50) #3 and 1000 work well for people
    node_colors = ['red' if s[-5] == 'R' else 'blue' for s in graph.nodes()]
    nx.draw_networkx_nodes(graph, my_plot, node_color = node_colors)
    edges_to_add = []
    for edge in graph.edges():
        if graph.get_edge_data(edge[0], edge[1])['weight'] > weight_cutoff:
            edges_to_add.append(edge)
    nx.draw_networkx_edges(graph, my_plot, edgelist = edges_to_add)
    if include_labels:
        labels = {}
        for i in graph.nodes(): labels[i] = i
        nx.draw_networkx_labels(graph, my_plot, labels)
    if save:
        plt.savefig('output/plots/{}.png'.format(title))
    if show:
        plt.show()
    plt.close('all')


def general_centrality_measures(graph):
    '''
    Gets centrality measures that are relatively quick for the people network and work for both directed and undirected 
    graphs.

    inputs:
    graph, a networkx graph object

    outputs:
    a list of centrality measurements
    '''
    degree_cent = nx.degree_centrality(graph) 
    pr = nx.pagerank(graph) 
    closeness = nx.closeness_centrality(graph)
    eigen = nx.eigenvector_centrality_numpy(graph, weight = "weight")
    load_cent = nx.load_centrality(graph, weight = "weight")
    return [degree_cent, pr, closeness, eigen, load_cent]


def all_centrality_scores(graph):
    '''
    Grabs all centrality scores for all nodes.  

    Inputs:
    graph, a networkx graph object

    Outputs:
    currently returns a dict person:[centrality_score1, centrality_score2, ...]
    '''
    df_dict = {}
    measures = ['degree_centrality', 'pagerank', 'closeness_centrality', 'eigenvector_centrality', 'load centrality']
    centrality_measures = general_centrality_measures(graph)
    for node in graph.nodes():
        df_dict[node] = [score_dict[node] for score_dict in centrality_measures]
    return df_dict


def construct_bill_network(master_dict):
    '''
    UNUSED -- centrality measurements took too long, so I cut it as impractical.
    I'm leaving it here so you can get a sense of one possible extension of my project.
    The initialization code is in the intialization function, but I do not integrate this functionality past that.

    Bills are nodes, edges if 2 bills sponsored/cosponsored by the same person
    Edge if any person sponsored/cosponsored both bills (edge between ever bill an individual sponsored/cosponsored)

    Example Network:

    Person 1 sponsored bills a and b 
    Person 2 sponsored bills a, b, and c 
    Person 3 sponsored bills a and c 

    3 nodes: a, b, and c 
    3 edges: (a-b, weight=2), (b-c weight=1), (a-c, weight=2)
    '''
    graph = nx.Graph()

    # format data for building graph
    d_by_person = {} 
    for bill in master_dict:
        if master_dict[bill][0] in d_by_person:
            d_by_person[master_dict[bill][0]].append(bill)
        else:
            d_by_person[master_dict[bill][0]] = [bill]
        for cosponsor in master_dict[bill][1]:
            if cosponsor in d_by_person:
                d_by_person[cosponsor].append(bill)
            else:
                d_by_person[cosponsor] = [bill]

    # get all edges, weighted by repetition.  Again, quicker than using update_network.
    edges = {}
    for person in d_by_person:
        for combo in combinations(d_by_person[person], 2):
            if combo in edges:
                edges[combo] += 1
             else:
                edges[combo] = 1

    # add all edges to graph
    for combo in edges:
      graph.add_edge(combo[0], combo[1], weight=edges[combo])
    return graph


def initialize_networks(init_dict):
    '''
    Initializing the networks with a dictionary that is a subset of all the bills.
    
    Input:
    init_dict, a dictionary that is a subset of master_dict

    Output:
    a tuple of both graphs
    '''
    undirected_people_graph = construct_people_network(init_dict, undirected = True)
    directed_people_graph = construct_people_network(init_dict, undirected = False)

    # The following works, but is only necessary to intialize the Bill network.  The output is not integrated into 
    # train_and_get_centrality_measures.

    # just_passed_bills = {}
    # for key in range(num_bills_init, len(init_dict)):
    #   if passage_dict[key] == 'passed':
    #       just_passed_bills[key] = master_dict[key]
    # bill_graph = construct_bill_network(just_passed_bills)

    return [directed_people_graph, undirected_people_graph]


def update_network(bill, bill_info, graph, undirected = False):
    '''
    Adds an individual bill to the people network.

    Inputs:
    bill, int, the bill number
    bill_info, a tuple/list len 2, [sponsor, [cosponsor1, cosponsor2 ...]]
    graph, a networkx graph object

    Output:
    an updated networkx graph object
    '''
    edges = {}
    for person in bill_info[1]:
        single_edge = (person, bill_info[0])
        if single_edge in graph.edges():
            graph[single_edge[0]][single_edge[1]]['weight'] += 1
        else:
            graph.add_edge(single_edge[0], single_edge[1], weight=1)
    return graph

def split_master_dict(master_dict, num_bills_init):
    '''
    Grabs the bills to initialize the network with.  

    Inputs:
    master_dict, the dictionary from the web scraper json 
    num_bills_init, the bill number before which to use in the intial network

    Output:
    init_dict, the dict to use in initialization
    '''
    init_dict = {}
    for key in master_dict:
        if key < num_bills_init:
            init_dict[key] = master_dict[key]
    return init_dict

def train_and_get_centrality_measures(num_bills_init = 100, visualize_every_n_bills = 500):
    '''
    Gives a dataframe that is each bill with its associated centrality metrics.

    Some bills are missing because they either weren't hosted, had nonsense texts (i.e. a bad file was uplaoded) or .
    '''
    # Load in data
    master_dict = open_convert_json('bill_sponsorships_final.json')
    passage_dict = open_convert_json('passage_data.json')

    # Temporal split
    init_dict = split_master_dict(master_dict, num_bills_init)

    # Initialize the graphs.  Currently only people dict directed and undirected
    list_of_graphs = initialize_networks(init_dict)
    print("{} bills are missing from the dataset".format(len(passage_dict) - len(master_dict)))
    print("Initialized networks with {} bills".format(len(init_dict)))
    print("{} bills will be added in".format(len(master_dict) - len(init_dict)))

    # Start adding in nodes and getting centrality measures from them.  Visualize every n bills, and at the end
    master_list_centrality_measures = []
    for bill in range(num_bills_init, len(passage_dict)):
        print("Added S.{} to the network".format(bill))
        try:
            list_for_each_bill = []
            for i, graph in enumerate(list_of_graphs):
                if bill % visualize_every_n_bills == 0:
                    visualize_graph(graph, weight_cutoff = bill/1000, title = "network2_after_bill_{}_{}".format(bill, i), save = True)
                    print("built {} with weight cutoffs at {}".format("network_after_bill_{}_{}.png".format(bill, i), bill/800))
                graph = update_network(bill, master_dict[bill], graph)
                cms = all_centrality_scores(graph)
                cosponsor_cms = []
                for cosponsor in master_dict[bill][1]:
                    cosponsor_cms.append(cms[cosponsor])
                if len(cosponsor_cms) == 0:
                    cosponsor_cms_avg = [0]*5
                else:
                    cosponsor_cms_avg = np.array(cosponsor_cms).mean(axis=0)
                sponsor_cms = cms[master_dict[bill][0]]
                list_for_each_bill += sponsor_cms + list(cosponsor_cms_avg)
                print(len(graph.nodes()))
            master_list_centrality_measures.append([bill, master_dict[bill][0], passage_dict[bill], len(master_dict[bill][1])] + list_for_each_bill)

        except KeyError:
            print("S.{} Not Here".format(bill))

    # Generate Outputs
    for i, graph in enumerate(list_of_graphs):
        visualize_graph(graph, title = "network2_after_all_bills{}".format(i), save = True)
    df = pd.DataFrame(master_list_centrality_measures)
    df.columns = ['Bill No.', 'Sponsor', 'did_it_pass', 'No. Cosponsors', 'dir_degree_centrality_sponsor', 'dir_pagerank_sponsor', 'dir_closeness_centrality_sponsor', \
                'dir_eigenvector_centrality_sponsor', 'dir_load_centrality_sponsor', 'dir_avg_degree_centrality_cosponsor', 
                'dir_avg_pagerank_cosponsor', 'dir_avg_closeness_centrality_cosponsor', 'dir_avg_eigenvector_centrality_cosponsor',
                'dir_avg_load_centrality_cosponsor', 'und_degree_centrality_sponsor', 'und_pagerank_sponsor', 'und_closeness_centrality_sponsor', \
                'und_eigenvector_centrality_sponsor', 'und_load_centrality_sponsor', 'und_avg_degree_centrality_cosponsor', 
                'und_avg_pagerank_cosponsor', 'und_avg_closeness_centrality_cosponsor', 'und_avg_eigenvector_centrality_cosponsor',
                'und_avg_load_centrality_cosponsor']
    return df # this is a dataframe of all my testing data

if __name__ == "__main__":
    df_mine = train_and_get_centrality_measures(100) # To test, I suggest setting it to ~4100. 
    df_mine.to_csv('output/network_with_senators.csv')

