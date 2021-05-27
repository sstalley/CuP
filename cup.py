import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

init_color="green"

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

    return (xx, yy)


# sampling function stolen from John's code
def sampleAt(loc, context, verbose=False):
    xx, yy = context
    if verbose:
        print("sampling at", loc)
    xbar = xx - 0.5
    ybar = yy - 0.5
    locbar = loc - 0.5
    rr = np.sqrt(xbar**2 + ybar**2)
    #tt = np.linspace(0, 2*np.pi, rr.shape[0])
    tt = 2*np.arctan(ybar / (xbar + rr))
    r = np.sqrt(locbar[0]**2 + locbar[1]**2)
    if (np.abs(locbar[1]) < 1e-8) and (locbar[0] > 0):
        tht = 0
    elif (np.abs(locbar[1]) < 1e-8) and (locbar[0] <= 0):
        tht = np.pi
    else:
        tht = 2*np.arctan(locbar[1] / (locbar[0] + r))
    ind = np.argmin(np.abs(tht - tt))
    if verbose:
        print("the bar is", rr[ind], "we are at", r)
    if rr[ind] < r:
        return 1.0
    else:
        return -1.0

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

    return G, err, dist, snaps, edges, pos
