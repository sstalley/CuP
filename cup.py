import math
import numpy as np
import networkx as nx
import matplotlib.path as mpltPath
import time
import geopandas
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry import LineString

init_color="green"
sampled_color = 'blue'

class grid_graph():
    def define_labels(self, G, nr, nc, boundary=None):
        path = mpltPath.Path(boundary)
        node_labels = dict(((i, j), 1 if path.contains_point(np.array([i/(nc-1), j/(nr-1)])) else 0) for i, j in G.nodes())
        color_map = dict(((i, j), 'yellow' if path.contains_point(np.array([i/(nc-1), j/(nr-1)])) else 'orange') for i, j in G.nodes())

        return node_labels, color_map
     
    def create_graph(self, w=10, boundary=None):
        nr = nc = w+1
        G = nx.grid_2d_graph(nr, nc)
        node_labels, color_map = self.define_labels(G, nr, nc, boundary=boundary)
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
def oracle_test(G, w, n_neighbors=4):
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

# GP on circle
def EXP_cov(x, y, l):
    diff  = np.subtract.outer(x, y)
    return np.exp(-2*(np.sin(diff/2)/l)**2)

def gen_gp_boundary(N=1000, mu=3, l=0.5, edge=20):
    theta = np.linspace(0,2*np.pi,N)
    SIG = EXP_cov(theta, theta, l)

    # generate random GP boundary
    rr = np.exp(np.random.multivariate_normal(mean=mu*np.ones(N), cov=SIG))
    xx = rr*np.cos(theta)
    xx = xx / (2*np.max(np.abs(xx)) + 1/edge) + 0.5
    yy = rr*np.sin(theta)
    yy = yy / (2*np.max(np.abs(yy)) + 1/edge) + 0.5

    boundary = np.empty((N, 2))
    boundary[:,0] = xx
    boundary[:,1] = yy

    return boundary

def cup2(G, w, init_plus, init_minus):
    snaps = 0
    edges = 0
    dist = 0.0

    rotate = np.array([[0, -1],[1,0]])
    node_labels = nx.get_node_attributes(G,'label')
    color_map = nx.get_node_attributes(G,'color')
    sample_map = nx.get_node_attributes(G,'sampled')

    assert node_labels[init_plus] == 1
    assert node_labels[init_minus] == 0

    color_map[init_plus]  = color_map[init_minus]  = init_color
    sample_map[init_plus] = sample_map[init_minus] = True
    snaps += 2
    dist += np.linalg.norm((np.array(init_plus) - np.array(init_minus))/w)
    G.remove_edge(init_plus, init_minus)
    edges +=1

    a = init_plus
    b = init_minus

    pos = source = init_plus
    dest = init_minus

    points = []
    points.append(np.array(pos)/w)
 
    # Until we have no path to follow
    while True:
        try:
            path = nx.shortest_path(G, source=source, target=dest)
        except nx.NetworkXNoPath:
            # if there is no path then a and b are elements of seperate components, then CuP is complete
            break

        # sample a cell until a new cut edge is found
        # note that the shortest path will find all the elements in the current cell (regardless of shape)
        for v in path[1:-1]: # ignore current pos and dest since they have already been sampled

            if sample_map[v] == False:
                dist += np.linalg.norm((np.array(v) - np.array(pos))/w) # only works for square grids
                pos = v
                points.append(np.array(pos)/w)
                snaps += 1
                sample_map[pos] = True

            if node_labels[v] == 1:
                a = source = v
                dest = b
            if node_labels[v] == 0:
                b = source = v
                dest = a
 
            if b in G.neighbors(a):
                # found an edge, remove it and move onto the next cell
                G.remove_edge(a, b)
                edges +=1
                break

    #TODO: error for non-square grids
    err = edges / (w * w)

    points = np.array(points)

    return G, err, dist, snaps, edges, pos, points

# edge - determines how close the GP gets to the edge of the graph
# N    - number of samples used in the radial gp # grid size for function
# mu   -  mean of the GP

def gen_test_boundary(modes, wRange):
    boundary = gen_gp_boundary(N=1000, mu=3, l=0.5, edge=10)

    if not check_boundary(boundary, wRange): # does not meet requirements
        return gen_test_boundary(modes, wRange)

    return boundary_tests(boundary, modes, wRange)

def check_boundary(boundary, wRange):
    # use oracle to see if this shape works before we run our algorithms
    for w in wRange:
        gg = grid_graph()
        G = gg.create_graph(boundary=boundary, w=w)
        try:
            G, area, mdist, nsamp = oracle_test(G, w)
        except ValueError:
            print(f"boundary doesn't meet requirements for w={w}")
            return False

    # it must work for every w then...
    return True

def boundary_tests(boundary, modes, wRange):

    ns    = np.zeros((len(wRange),  len(modes)))
    dist  = np.zeros((len(wRange),  len(modes)))
    err   = np.zeros((len(wRange),  len(modes)))
    rtime = np.zeros((len(wRange),  len(modes)))

    for ww, w in enumerate(wRange):
        maxiter = w * w * 2 #should finish well before it goes everywhere twice

        # run search
        for m, mode in enumerate(modes):
            
            gg = grid_graph()
            G = gg.create_graph(boundary=boundary, w=w)
            init_plus, init_minus = find_init(G)
            a0 = np.array(init_plus)/w
            b0 = np.array(init_minus)/w

            start = time.time()

            if mode == 'CuP2':
                G, area, mdist, nsamp, _, _, _ = cup2(G, w, init_plus, init_minus)

            elif mode == 's2':
                G, area, mdist, nsamp = s2(G, maxiter, w, init_plus, init_minus)

            elif mode == 'oracle':
                G, area, mdist, nsamp = oracle_test(G, w)
   
            rt = time.time() - start 
            print(mode, w, ": Samples:", nsamp, "Distance:", mdist, "Error:", area, "Runtime:", rt)

            ns[ww, m] = nsamp
            dist[ww, m] = mdist
            err[ww, m] = area
            rtime[ww,m] = rt

    return ns, dist, err, rtime


def gframe_from_file(sfile, maxacres, statestr="OR"):
    gframe = geopandas.GeoDataFrame.from_file(sfile)
    gframe = gframe[gframe['Event_ID'].str.contains(statestr)]
    gframe = gframe[gframe['BurnBndAc'] <= maxacres]
    #Only consider contiguous fires
    gframe = gframe[gframe['geometry'].apply(lambda x: isinstance(x, Polygon))]
    return gframe

def scale_boundaries(gframe):
    longscale = 69.172 * 1.609 * np.cos(44.15) # approximated to the scale in the center of oregon
    latscale = 69.0 * 1.609
    maxrange = 0.0

    for (i, fire) in gframe.iterrows():
        (xmin, ymin, xmax, ymax) = fire['geometry'].bounds
        fmaxrange = max((xmax - xmin) * longscale, (ymax - ymin) * latscale)
        if fmaxrange > maxrange:
            maxrange = fmaxrange
    
    maxrange = math.ceil(maxrange) 
    print("max range (in Km):", maxrange)
    
    for (i, fire) in gframe.iterrows():
        # save the scaled geometry for CuP
        (xmin, ymin, xmax, ymax) = fire['geometry'].bounds
        xoff = 0.5 - ((xmax + xmin)/2)
        yoff = 0.5 - ((ymax + ymin)/2)
        xfact = longscale/maxrange
        yfact = latscale/maxrange

        gframe.at[i, 'xoff']      = xoff
        gframe.at[i, 'yoff']      = yoff
        gframe.at[i, 'xfact']     = xfact
        gframe.at[i, 'yfact']     = yfact
        gframe.at[i, 'sgeometry'] = affinity.scale(affinity.translate(fire['geometry'], xoff=xoff, yoff=yoff), xfact=xfact, yfact=yfact, origin=(0.5, 0.5))
    #print("sgeometry bounds:", fire['sgeometry'].bounds)

    return gframe

def calc_save_sbs(gframe, sbname):
    sbs = []

    for (i, fire) in gframe.iterrows():
        sb = np.array(fire['sgeometry'].exterior.coords.xy).T

        print("sb:", sb)
        sbs.append(sb)

    np.save(sbname, sbs)    
    return sbs

def map_paths(gframe, paths):

    linepaths = []
    for (i, fire), path in zip(gframe.iterrows(), paths):
        linepath = LineString(np.array(path))
        linepath = affinity.translate(
                                      affinity.scale(linepath, xfact=fire.xfact, yfact=fire.yfact, origin=(0.5, 0.5)),
                                      xoff=fire.xoff, yoff=fire.yoff)
        linepaths.append(linepath)
    
    mapped_paths = geopandas.GeoDataFrame(geometry=linepaths, crs="EPSG:4326")
    return mapped_paths

def unscale_paths(gframe, paths):
    us_paths = []

    for (i, fire), path in zip(gframe.iterrows(), paths):
        us_path = np.empty_like(path)
        us_path[:,1] = (path[:,0] - 0.5)/fire['xfact'] + 0.5 - fire['xoff']
        us_path[:,0] = (path[:,1] - 0.5)/fire['yfact'] + 0.5 - fire['yoff']

        print("fire", i, "first point location:", us_path[0,:])

        us_paths.append(us_path)

    return us_paths
