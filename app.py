import streamlit as st
import pandas as pd
import numpy as np

import sklearn.neighbors

import pydeck as pdk
import seaborn as sns

from util import config
from util import mapping
from util import trip_data

@st.cache
def load_data():
    data = pd.read_csv(config.PROCESSED_DATA_PATH + 'trips.csv', index_col=0)
    # df = filter_cleaned_trips(df)

    rts_gridpts = pd.read_csv(config.PROCESSED_DATA_PATH + 'rts_grid_pts.csv', index_col=0)
    rts_gridpts.grid_pts = rts_gridpts.grid_pts.apply(lambda x: [int(a) for a in x.strip('[]').split(',')])

    grid_pts_fine = pd.read_csv(config.MODEL_PATH + 'road_backbone_merged.csv', index_col=0)

    return data, grid_pts_fine, rts_gridpts

def load_coarse_grid():
    grid_pts, grid_dict = trip_data.load_gridpts(
        'road_backbone_coarse', 'grid_rte_ids_coarse'
    )
    loc_tree = sklearn.neighbors.KDTree(grid_pts[['lat', 'lon']])
    return grid_pts, grid_dict, loc_tree

@st.cache
def load_presets():
    presets, presets_labels = trip_data.set_presets()
    presets = trip_data.apply_scaling(presets)
    return (presets, presets_labels)

@st.cache(allow_output_mutation=True)
def fit_tree(df, feature_importance):
    LEAF_SIZE = 20
    return sklearn.neighbors.KDTree(df * feature_importance, leaf_size=LEAF_SIZE)


def calc_dist_point_to_grid(start_lat, start_lon, grid_pts, grid_dict, loc_tree):
    dists, inds = loc_tree.query(np.array([[start_lat, start_lon]]), k=2000)
    dists *= mapping.degrees_to_miles_ish(1)
    rtes_done = set()
    ds = []
    for dist_to_point, i in zip(dists.ravel(), inds.ravel()):
        grid_id = grid_pts.at[i, 'grid_i']
        rtes_at_point = grid_dict[grid_id].copy()
        rtes_at_point -= rtes_done
        ds += [{'dist_to_start': dist_to_point, 'id': rte} for rte in rtes_at_point]
        rtes_done = rtes_done.union(rtes_at_point)

    return pd.DataFrame(ds)

def cat_distance(x):
    if x < 1:
        return 0
    if x < 2:
        return 0.5
    if x < 5:
        return 3
    else:
        return 10

data, grid_pts_fine, rts_gridpts = load_data()
feature_importance = [1] * data.shape[1]
tree = fit_tree(data, feature_importance)
grid_pts, grid_dict, loc_tree = load_coarse_grid()

# Set up ride style choice:
st.title('It\'s the Catskills!')
st.subheader('Go to the sidebar to personalise your ride suggestion!')
st.sidebar.subheader('Riders! What are you in the mood for?')
start_location_yn = st.sidebar.checkbox('Specify start location?')
if start_location_yn:
    MAX_DIST_FROM_START = 10 # miles
    start_lat = st.sidebar.text_input('Latitude (N):', 42)
    start_lon = st.sidebar.text_input('Longitude (W):', 74.25)
    start_lon = float(start_lon) * -1 # convert to degrees east
    start_lat = float(start_lat)
    # st.markdown(float(start_lat))
    # st.markdown(-float(start_lon))
    dist_to_point = calc_dist_point_to_grid(start_lat, start_lon,
                                            grid_pts, grid_dict, loc_tree)
    data_loc = pd.merge(data, dist_to_point, on='id', how='inner').set_index('id')
    data_loc = data_loc[data_loc['dist_to_start'] < MAX_DIST_FROM_START]
    unscaled_dists = data_loc[['dist_to_start']].copy()
    data_loc['dist_to_start'] = data_loc['dist_to_start'].apply(cat_distance)
    tree = fit_tree(data_loc, feature_importance + [1])
presets, presets_labels = load_presets()
self_enter_lab = 'Enter my own'
option = st.sidebar.selectbox('Pick one:', [self_enter_lab] + presets_labels)

if option == self_enter_lab:
    st.markdown('Please enter your preferred ride parameters in the sidebar, or select one of our preset options.')
    dist_in = st.sidebar.slider('Distance:', min_value=5., max_value=100., value=20., step=2.5)
    slope_in = st.sidebar.slider('Average slope:', min_value=1., max_value=10., value=2., step=0.5)
    speed_in = st.sidebar.slider('Average speed:', min_value=6., max_value=20., value=10., step=2.)
    breaks_in = st.sidebar.select_slider('Opportunity for breaks:', ['low', 'medium', 'high'], value='medium')

    breaks_dict = {'low': 1.0, 'medium': 0.85, 'high': 0.7}
    breaks_in = breaks_dict[breaks_in]

    chosen = pd.DataFrame([[dist_in, slope_in, speed_in, breaks_in]],
                        columns=presets.columns)
    chosen
    if chosen.iloc[0].isna().sum():
        st.markdown('Please fill in all fields!')
    chosen = trip_data.apply_scaling(chosen)
else:
    ind = presets_labels.index(option)
    chosen = presets.loc[[ind]]

if not chosen.iloc[0].isna().sum():
    N_RIDES = 5
    if start_location_yn:
        chosen['dist_to_start'] = 0
        data_use = data_loc
    else:
        data_use = data
    dist, ind = tree.query(chosen, k=N_RIDES)

    colours = sns.color_palette('husl', N_RIDES)
    ride_layers = []
    map_lims = np.array([[90, -90, 180, -180]])
    d = trip_data.remove_scaling(data_use.iloc[ind[0]])
    if start_location_yn:
        d['dist_to_start'] = unscaled_dists.iloc[ind[0]]
    d
    for i in range(N_RIDES):
        ride_id = data_use.index[ind[0, i]]
        ride_points = grid_pts_fine.loc[rts_gridpts.loc[ride_id].grid_pts][['lat', 'lon']]
        ride_layers += [
            pdk.Layer(
                type="ScatterplotLayer",
                data=pd.DataFrame({
                    'name': ['{}'.format(ride_id)] * ride_points.shape[0],
                    'coordinates': [[row.lon, row.lat] for i, row in ride_points.iterrows()],
                    'size': [1] * ride_points.shape[0],
                }),
                opacity= 1,
                pickable=True,
                filled=True,
                stroked=False,
                radius_min_pixels=1,
                get_position="coordinates",
                get_radius="size",
                get_fill_color=[int(x * 255) for x in colours[i]],
            )
        ]
        map_lims = np.vstack((map_lims,
                            np.array([[ride_points.lat.min(),
                                       ride_points.lat.max(),
                                       ride_points.lon.min(),
                                       ride_points.lon.max()]])))
        # st.markdown('{},{} - {},{}'.format(min_lat, min_lon, max_lat, max_lon))

    max_lat, max_lon = tuple(map_lims[:, 1:4:2].max(axis=0))
    min_lat, min_lon = tuple(map_lims[:, 0:3:2].min(axis=0))

    if start_location_yn:
        df = pd.DataFrame({'name': ['start point'], 'size': [2],
                            'coordinates': [[start_lon, start_lat]]})
        ride_layers += [pdk.Layer(
                type="ScatterplotLayer",
                data=df,
                opacity= 1,
                filled=True,
                stroked=False,
                radius_min_pixels=3,
                get_position="coordinates",
                get_radius="size",
                get_fill_color=[255, 0, 0],
        )]

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=pdk.data_utils.compute_view([
            [max_lon + 0.25, max_lat + 0.25],
            [max_lon + 0.25, min_lat - 0.25],
            [min_lon - 0.25, max_lat + 0.25],
            [min_lon - 0.25, min_lat - 0.25],
        ]),
        layers=ride_layers,
        tooltip={"html": "<b>Ride {name}</b>", "style": {"color": "white"}},
    ))
else:
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude": 41.979, "longitude": -74.218, "zoom":9},
        #layers=ride_layers,
    ))
