import numpy as np
import matplotlib.pyplot as plt
import cup
import s2
import time
from joblib import Parallel, delayed


# initial parameters
saveFigs = True
MC = 100
modes = ['CuP2', 's2']
mode_labels = ['CuP', '$S^{2}$']
wRange = range(10,50+1)
edge = 10 # determines how close the GP gets to the edge of the graph
N = 1000 # number of samples used in the radial gp # grid size for function
mu = 3 # mean of the GP
l = 0.5 
graph_type = 'radial-gp'
figpath ='../papers/cupSPL/Figures/'

ylabels = ['number of samples', 'distance traveled', 'estimation error', 'runtime (s)']
fnames  = ['sampVsW.eps',       'distVsW.eps',       'errVsW.eps',       'runtimeVsW.eps']

def boundry_tests(modes, edge, N, mu):

    ns    = np.zeros((len(wRange),  len(modes)))
    dist  = np.zeros((len(wRange),  len(modes)))
    err   = np.zeros((len(wRange),  len(modes)))
    rtime = np.zeros((len(wRange),  len(modes)))

    (xx, yy) = cup.gen_gp_boundary(N=N, mu=mu, l=l, edge=edge)

    context = (xx, yy)

    # use oracle to see if this shape works before we run our algorithms
    for w in wRange:
        gg = s2.grid_graph()
        G = gg.create_graph(graph_type=graph_type, context=context, w=w)
        try:
            G, area, mdist, nsamp = s2.oracle_test(G, w, context)
        except ValueError:
            print(f"GP doesn't meet requirements for w={w}, re-rolling...")
            return boundry_tests(modes, edge, N, mu)

    for ww, w in enumerate(wRange):
        maxiter = w * w * 2 #should finish well before it goes everywhere twice

        # run search
        for m, mode in enumerate(modes):
            
            gg = s2.grid_graph()
            G = gg.create_graph(graph_type=graph_type, context=context, w=w)
            init_plus, init_minus = s2.find_init(G)
            a0 = np.array(init_plus)/w
            b0 = np.array(init_minus)/w

            start = time.time()

            if mode == 'CuP2':
                G, area, mdist, nsamp, _, _ = cup.cup2(G, w, init_plus, init_minus)

            elif mode == 's2':
                G, area, mdist, nsamp = s2.s2(G, maxiter, w, init_plus, init_minus)

            elif mode == 'oracle':
                G, area, mdist, nsamp = s2.oracle_test(G, w, context)
   
            rt = time.time() - start 
            print(mode, w, ": Samples:", nsamp, "Distance:", mdist, "Error:", area, "Runtime:", rt)

            ns[ww, m] = nsamp
            dist[ww, m] = mdist
            err[ww, m] = area
            rtime[ww,m] = rt

    return ns, dist, err, rtime


cache_name = "./both_cache_MC" + str(MC) + ".npy"

try:
    print("trying to load cached values from", cache_name)
    retnp = np.load(cache_name)
    print("loaded cached values")

except IOError:
    print("could not load cache, calculating clusters")
    ret = Parallel(n_jobs=23, verbose=10)(delayed(boundry_tests)(modes, edge, N, mu) for mc in range(MC))
    retnp = np.array([np.array(xi) for xi in ret])
    np.save(cache_name, retnp)    


for i, (ylabel, fname) in enumerate(zip(ylabels, fnames)):
    plt.rc('text',usetex=True)
    plt.rc('font',family='serif')
    plt.figure()
    plt.plot(wRange, np.mean(retnp[:, i, :, :], axis=0))
    plt.tick_params(labelsize=20)
    plt.xlabel(r'inverse side length ($w$)', fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.legend(mode_labels, fontsize=24)
    plt.tight_layout()
    if saveFigs:
        plt.savefig(figpath + fname, dpi=300)
    else:
        plt.show()
    plt.cla()
