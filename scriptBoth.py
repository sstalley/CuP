import numpy as np
import matplotlib.pyplot as plt
import cup
from joblib import Parallel, delayed
import folium

# initial parameters
saveFigs = True
MC = 100
#MC = 10 # FOR TESTING
modes = ['CuP2', 's2']
mode_labels = ['CuP', '$S^{2}$']
wRange = range(10,50+1)
figpath ='../papers/cupSPL/Figures/'
sbname="./sbs.npy"

#stuff for the fire maps
wRange2 = [40, 50, 60, 70, 80, 90, 100]
sfile = "./mtbs/mtbs_perims_DD.shp"
maxacres = 5000
statestr="OR"

def plot_results(retnp, ylabel, fname, wRange):
    plt.rc('text',usetex=True)
    plt.rc('font',family='serif')
    plt.figure()
    plt.plot(wRange, np.mean(retnp, axis=0))
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


ylabels = ['number of samples', 'distance traveled', 'estimation error', 'runtime (s)']
fnames  = ['sampVsW.eps',       'distVsW.eps',       'errVsW.eps',       'runtimeVsW.eps']
fnames2 = ['sampVsWfire.eps',   'distVsWfire.eps',   'errVsWfire.eps',   'runtimeVsWfire.eps']

#Scale to real dimensions
ylabels2 = ['number of samples', 'distance traveled (km)', 'estimation error (km$^2$)', 'runtime (s)']
scales =   [                  1,                       20,                       20*20,             1]

cache_name = "./both_cache_MC" + str(MC) + ".npy"
cache_name2 = "./both_cache2.npy"

try:
    print("trying to load cached values from", cache_name)
    retnp = np.load(cache_name)
    print("loaded cached values")

except IOError:
    print("could not load cache, running simulation")
    ret = Parallel(n_jobs=23, verbose=10)(delayed(cup.gen_test_boundary)(modes, wRange) for mc in range(MC))
    retnp = np.array([np.array(xi) for xi in ret])
    np.save(cache_name, retnp)

for i, (ylabel, fname) in enumerate(zip(ylabels, fnames)):
    plot_results(retnp[:, i, :, :], ylabel, fname, wRange)

try:
    print("trying to load cached values from", cache_name2)
    retnp2 = np.load(cache_name2)
    print("loaded cached values")

except IOError:
    print("could not load cache, running simulation")
    try:
        print("trying to load fire boundaries from", sbname)
        sbs = np.load(sbname, allow_pickle=True)
        print("trying to load database at", sfile)
        gframe = cup.gframe_from_file(sfile, maxacres, statestr)
        print("loaded fire boundaries")
    except IOError:
        print("could not load fire boundaries, recomputing boundaries")
        try:
            gframe = cup.gframe_from_file(sfile, maxacres, statestr)
        except IOError:
            print("could not load fire database from", sfile)
            quit()

        gframe = cup.scale_boundaries(gframe)
        sbs = cup.calc_save_sbs(gframe, sbname)

    ret2 = Parallel(n_jobs=23, verbose=10)(delayed(cup.check_boundary)(sb, wRange2) for sb in sbs)
    keepers = np.array([np.array(xi) for xi in ret2])
    sbs_keepers = sbs[keepers]
    gframe_keepers = gframe[keepers]
    print("gframe_keepers:", gframe_keepers)
    print("len(sbs_keepers) = ", len(sbs_keepers))
    retnp2 = Parallel(n_jobs=23, verbose=10)(delayed(cup.boundary_tests)(sb, modes, wRange2) for sb in sbs_keepers)
    retnp2 = np.array([np.array(xi) for xi in retnp2])

    np.save(cache_name2, retnp2)    

print("fire complete, saving results")

for i, (ylabel, fname, scale) in enumerate(zip(ylabels2, fnames2, scales)):
    plot_results(retnp2[:, i, :, :]*scale, ylabel, fname, wRange2)
