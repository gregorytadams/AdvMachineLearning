import json
from pprint import pprint
from networkx.readwrite import json_graph
import networkx

def get_force_graph(congress):
    '''
    This function gets the data that is displayed in our network visualization.

    Input: congress number 

    Output: three graphs, one for each the total senate, republican and democratic parties
    '''
    file_name = 'gov_data/static/gov_data/force_json/'+str(congress)+'_force.json'
    json_data=open(file_name).read()
    data = json.loads(json_data)
    graph = json_graph.node_link_graph(data)
    republican_graph = json_graph.node_link_graph(data)
    democrat_graph = json_graph.node_link_graph(data)

    return graph, republican_graph, democrat_graph

def create_party_graph(congress):
    '''
    Creates the actual datastructure that is displayed and saves it.
    '''
    republican_list = []
    democrat_list = []
    other_list = []
    
    graph, republican_graph, democrat_graph = get_force_graph(congress)
    
    party=networkx.get_node_attributes(graph,'party')
    
    for node in graph.nodes():
        try:
            if party[node] == 'r':
                republican_list.append(node)
            elif party[node] == 'd':
                democrat_list.append(node)
            else:
                other_list.append(node)
        except:
            other_list.append(node)

    republican_graph.remove_nodes_from(democrat_list)
    republican_graph.remove_nodes_from(other_list)

    democrat_graph.remove_nodes_from(republican_list)
    democrat_graph.remove_nodes_from(other_list)

    republican_data = json_graph.node_link_data(republican_graph)
    democrat_data = json_graph.node_link_data(democrat_graph)

    json.dump(republican_data, open('gov_data/static/gov_data/party_json/'+ str(congress) + '_rep_force.json','w'))
    json.dump(democrat_data, open('gov_data/static/gov_data/party_json/'+ str(congress) + '_dem_force.json','w'))

if __name__ == '__main__':
    for congress in range(92, 115):
        create_party_graph(congress)
