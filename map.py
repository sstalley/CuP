import numpy as np
import matplotlib.pyplot as plt
import cup
from joblib import Parallel, delayed
import folium

sfile = "./mtbs/mtbs_perims_DD.shp"
sbname="./sbs.npy"
maxacres = 5000
statestr="OR"
w = 50
wRange = [w]
first_fire="STALEY"

def save_map(gframe, fname="testmap.html", paths=None, markers=False):
    gjson = gframe.drop(columns=["sgeometry"]).to_crs(epsg='4326').to_json()

    initgframe = gframe[gframe['Incid_Name'].str.contains(first_fire)]

    init_loc= (initgframe.iloc[0]['BurnBndLat'], initgframe.iloc[0]['BurnBndLon'])

    fmap  = folium.Map(init_loc,
                       zoom_start=14,
                       zoom_control=False,
                       tiles='cartodbpositron')
    fires = folium.features.GeoJson(gjson)
    tile  = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(fmap)

    if markers:
        for (i, fire) in gframe.iterrows():
            marker = folium.Marker(location=[fire['BurnBndLat'], fire['BurnBndLon']],
                tooltip=fire['Incid_Name'] + " " + fire["Ig_Date"] + " (" + str(fire['BurnBndAc']) + " acres)").add_to(fmap)

    if paths is not None:
        for path in paths:
            marker = folium.PolyLine(path, color='red').add_to(fmap)

    fmap.add_child(fires)
    fmap.save(fname)

def get_path(sb, w):
    gg = cup.grid_graph()
    G = gg.create_graph(boundary=sb, w=w)
    init_plus, init_minus = cup.find_init(G)
    _, _, _, _, _, _, point = cup.cup2(G, w, init_plus, init_minus)
    return point


sbs = np.load(sbname, allow_pickle=True)
gframe = cup.gframe_from_file(sfile, maxacres, statestr)
gframe = cup.scale_boundaries(gframe)

ret2 = Parallel(n_jobs=23, verbose=10)(delayed(cup.check_boundary)(sb, wRange) for sb in sbs)
keepers = np.array([np.array(xi) for xi in ret2])
 
sbs_keepers = sbs[keepers]
gframe_keepers = gframe[keepers]
points = Parallel(n_jobs=23, verbose=10)(delayed(get_path)(sb, w) for sb in sbs_keepers)


unscaled_paths = cup.unscale_paths(gframe_keepers, points)

save_map(gframe_keepers, paths=unscaled_paths)

print("len(sbs_keepers):", len(sbs_keepers))
print("len(sbs):", len(sbs))

