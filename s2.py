import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cup

sampled_color = 'blue'

class grid_graph():
    def define_labels(self, G, nr, nc, category, context=None):
        if category == 'linear-seperable':
            node_labels = dict(((i, j), 1 if i < nc/2 else 0) for i, j in G.nodes())
            color_map = dict(((i, j), 'yellow' if i < nc/2 else 'orange') for i, j in G.nodes())
        elif category == 'box-in-a-box':
            node_labels = dict(((i, j), 1 if ((i >= 3)and(i <= 5)) and ((j >= 3) and (j <= 5))  else 0) for i, j in G.nodes())
            color_map = dict(((i, j), 'yellow' if ((i >= 3)and(i <= 5)) and ((j >= 3) and (j <= 5)) else 'orange') for i, j in G.nodes())
        elif category == 'plus':
            node_labels = dict(((i, j), 1 if i == 1 or j == 1 else 0) for i, j in G.nodes())
            color_map = dict(((i, j), 'yellow' if i == 1 or j == 1 else 'orange') for i, j in G.nodes())
        elif category == 'radial-gp':
            node_labels = dict(((i, j), 1 if cup.sampleAt(np.array([i/(nc-1), j/(nr-1)]), context) == 1.0 else 0) for i, j in G.nodes())
            color_map = dict(((i, j), 'yellow' if cup.sampleAt(np.array([i/(nc-1), j/(nr-1)]), context) == 1.0 else 'orange') for i, j in G.nodes())

        return node_labels, color_map
     
    def create_graph(self, graph_type, w=10, context=None):
        nr = nc = w+1
        G = nx.grid_2d_graph(nr, nc)
        node_labels, color_map = self.define_labels(G, nr, nc, graph_type, context=context)
        pos = dict((n, n) for n in G.nodes())
        for (i,j) in list(node_labels):
            G.nodes[(i,j)]['pos'] = pos[(i, j)]
            G.nodes[(i,j)]['label'] = node_labels[(i, j)]
            G.nodes[(i,j)]['color'] = color_map[(i, j)]
            G.nodes[(i,j)]['sampled'] = False
        return G

def find_init(G):
    node_labels = nx.get_node_attributes(G,'label')
    
    for i, j in G.nodes():
        for k,l in G.neighbors((i, j)):
            if node_labels[(i, j)] != node_labels[(k,l)]:
                if node_labels[(i, j)] == 1:
                    return (i, j), (k, l)
                else:
                    return (k, l), (i, j)

# finds the number of edges and minimum number of samples
def oracle_test(G, w, context, n_neighbors=4):
    node_labels = nx.get_node_attributes(G,'label')
    color_map = nx.get_node_attributes(G,'color')
    edges = 0
    snaps = 0
    
    for i, j in G.nodes():
        for k,l in list(G.neighbors((i, j))):
            if node_labels[(i, j)] != node_labels[(k,l)]:
                G.remove_edge((i,j), (k,l))
                edges += 1
                color_map[(i,j)] = sampled_color
                color_map[(k,l)] = sampled_color

    ncomp = nx.number_connected_components(G)
    if ncomp > 2:
        raise ValueError(f"expected 2 connected components, but graph contains {ncomp}.")

    for i, j in G.nodes():
        if color_map[(i,j)] == sampled_color:
            snaps += 1

    xx, yy = context
    mdist = (snaps - 1) / w # assumes that a path exists where every sample taken is next to boundary

    area = edges / w / w
    return G, area, mdist, snaps

def s2(G, budget, w, init_plus, init_minus):
    snaps = 0
    edges = 0
    dist = 0.0
    pos = nx.get_node_attributes(G,'pos')
    node_labels = nx.get_node_attributes(G,'label')
    color_map = nx.get_node_attributes(G,'color')
    sampled_nodes = [[],[]]

    assert node_labels[init_plus] == 1 and  node_labels[init_minus] == 0
    sampled_nodes[1].append(init_plus)
    sampled_nodes[0].append(init_minus)
    color_map[init_plus] = 'green' 
    color_map[init_minus] = 'green'
    snaps += 2
    dist += np.linalg.norm((np.array(init_plus) - np.array(init_minus))/w)
    node1 = init_minus
    
    if budget > len(node_labels):
        budget = len(node_labels)

    while (len(sampled_nodes[0]) + len(sampled_nodes[1]) < budget) and (nx.number_connected_components(G) < 2):

        min_length = G.number_of_nodes()
        min_path = []
        min_travel_len = 1000 #any value > sqrt(2) should work

        for cutoff in range(min_length): # should stop well before min_length
            for node_plus in sampled_nodes[1]:
                paths = nx.single_source_shortest_path(G, node_plus, cutoff=cutoff)
                for dest in paths.keys():
                    if dest in sampled_nodes[0]: # we found one
                        path = paths[dest]
                        length = len(path) - 1
                        node_index = int(length/2)
                        node2 = path[node_index]
                        # print("found viable path of length", length, "from", node_plus, "to", dest)
                        travel_len = np.linalg.norm((np.array(node1) - np.array(node2))/w)
                        if travel_len < min_travel_len and length <= min_length:
                            min_length = length
                            min_path = path 
                            min_travel_len = travel_len
                            min_node_index = node_index
                            break

                        if length % 2 == 1: # odd number of edges => even number of nodes, so check the "other middle" node
                            node_index = int(length/2)+1
                            node2 = path[node_index]
                            travel_len = np.linalg.norm((np.array(node1) - np.array(node2))/w)
                            if travel_len < min_travel_len and length <= min_length:
                                # print("used other middle path of length", length, "from", node_plus, "to", dest)
                                min_length = length
                                min_path = path 
                                min_travel_len = travel_len
                                min_node_index = node_index

                if (len(min_path) > 0): # no need to check other other paths if we already found a shortest-shortest
                    break
            if len(min_path) > 0: 
                # print("found a shortest-shortest path, breaking")
                break

        if min_length == 1:
            #print("cutting edge")
            G.remove_edge(min_path[0], min_path[-1])
            edges +=1
        else:
            new_node = min_path[min_node_index]
            # Asumming a square grid
            travel = np.linalg.norm([(node1[0] - new_node[0])/w, (node1[1] - new_node[1])/w])
            #print("traveling", travel)
            dist +=travel
            node1 = new_node
            #print("sampling at", node1)
            sampled_nodes[node_labels[node1]].append(node1)
            color_map[node1] = sampled_color
            snaps += 1

    # Asumming a square grid
    err = edges / (w * w)
    # print("found ", edges, "cut edges with", snaps, "samples and traveled", dist)
    # print("boundry error", err)

    return G, err, dist, snaps
