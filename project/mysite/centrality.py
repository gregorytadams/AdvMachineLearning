# Degree Centrality
#
# Given a JSON file containing data about a graph, applies the degree centrality
# algorithm to determine who the most influential leaders in Senate are.


import networkx as nx
from networkx.readwrite import json_graph
import json

def centrality_analysis(filename):
    '''
    Opens and analyzes a json file to estimate centrality.

    input: json filename

    ouput: dict with centrality values
    '''
    with open(filename) as f:
        js_graph = json.load(f)
    G = json_graph.node_link_graph(js_graph)
    connectedness = nx.degree_centrality(G)
    influence = nx.eigenvector_centrality(G) #need to do error exception thingamajigs
    senator_centrality = {}
    for s_id in connectedness:
        senator_centrality[s_id] = {}
        senator_centrality[s_id]['Influence'] = influence[s_id]
        senator_centrality[s_id]['Connectedness'] = connectedness[s_id]
    return senator_centrality


if __name__ == '__main__':
    start = 110
    for congress in range(start, 114):
        filename = 'gov_data/static/gov_data/'+str(congress) +'_force.json'
        senator_centrality = centrality_analysis(filename)
        print("Finished congress {}".format(congress))
