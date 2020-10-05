import json
import collections
import pandas as pd
import numpy as np

from util import config


def set_presets():
    df = pd.read_csv(config.PROCESSED_DATA_PATH + 'trips.csv')

    presets_descriptions = [
        'Chilling out in the saddle', 'Pretty relaxed, with some climbing',
        'Half-day of touring', 'Training for VO2-max', 'Training for strength',
        'Training for a century']

    presets = pd.DataFrame({
        'dist': [10., 15., 45., 20., 10., 85.],
        'avg_slope_climbing': [3., 6., 5., 5., 8., 4.],
        'avg_slope_descending': [-3., -6., -5., -5., -8., -4.],
        'max_slope': [6., 10., 10., 6., 15., 10.],
        'dist_climbing': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'dist_downhill': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'dist_6percent': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'dist_9percent': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        'dist_12percent': [0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
        'detour_score': [10, 10, 10, 10, 10, 10],
        'popularity': [10, 10, 10, 10, 10, 10],
    })

    return presets, presets_descriptions

def engineer_features(df):

    df_eng = df.copy()
    df_eng['dist'] = np.log(df.dist +1e-2)
    df_eng['dist_6percent'] = np.log(df.dist_6percent + 1e-2)
    df_eng['dist_9percent'] = np.log(df.dist_9percent + 1e-2)
    df_eng['dist_12percent'] = np.log(df.dist_12percent + 1e-2)

    # df_eng.to_feather(config.PROCESSED_DATA_PATH + 'trips_eng.feather')
    return df_eng

def reverse_engineer_features(df):

    df_reverse = df.copy()
    df_reverse['dist'] = np.exp(df.dist) - 1e-2
    df_reverse['dist_6percent'] = np.exp(df.dist_6percent) - 1e-2
    df_reverse['dist_9percent'] = np.exp(df.dist_9percent) - 1e-2
    df_reverse['dist_12percent'] = np.exp(df.dist_12percent) - 1e-2

    return df_reverse


def scale_dataset(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = sklearn.preprocessing.StandardScaler()
    cols = df.columns.tolist()
    if 'rte_id' in cols:
        cols.remove('rte_id')

    df[cols] = scaler.fit_transform(df[cols])

    scaler_df = pd.DataFrame(np.vstack((scaler.mean_, scaler.scale_)),
                     columns=cols, index=['mean', 'std'])
    scaler_df.reset_index().to_feather(config.MODEL_PATH + 'feature_scaling.feather')

    df.to_feather(config.PROCESSED_DATA_PATH + 'trips_scaled.feather')

    return df

def remove_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_feather(config.MODEL_PATH + 'feature_scaling.feather')
    scaler = scaler.set_index('index')
    df_unscale = df.copy()
    for col in scaler.columns:
        df_unscale[col] = (df_unscale[col]
                    * scaler.loc['std', col] + scaler.loc['mean', col])

    return df_unscale

def apply_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_feather(config.MODEL_PATH + 'feature_scaling.feather')
    scaler = scaler.set_index('index')
    df_scale = df.copy()
    for col in scaler.columns:
        df_scale[col] = ((df_scale[col] - scaler.loc['mean', col])
                            / scaler.loc['std', col])

    return df_scale


def add_distance_to_start_feature(lat, lon, trips_df, grid_pts, rtes_at_grid,
                                  loc_tree, max_dist_from_start=10):

    dist_to_point = calc_dist_from_point_to_rtes(lat, lon, grid_pts, rtes_at_grid, loc_tree)
    trips_loc = pd.merge(trips_df, dist_to_point, on='rte_id', how='inner').set_index('rte_id')
    # Filter out routes that are too close
    trips_loc = trips_loc[trips_loc['dist_to_start'] < max_dist_from_start] # miles
    unscaled_dists = trips_loc[['dist_to_start']].copy()
    trips_loc['dist_to_start'] = trips_loc['dist_to_start'].apply(bin_ride_distance, args=[max_dist_from_start])

    return trips_loc, unscaled_dists

def calc_dist_from_point_to_rtes(start_lat, start_lon, grid_pts, rtes_at_grid, loc_tree):
    dists, inds = loc_tree.query(np.array([[start_lat, start_lon]]), k=2000)
    dists *= mapping.degrees_to_miles_ish(1)
    rtes_done = set()
    ds = []
    for dist_to_point, i in zip(dists.ravel(), inds.ravel()):
        grid_id = grid_pts.index[i]
        rtes_at_point = set(rtes_at_grid.loc[grid_id].rte_ids)
        rtes_at_point -= rtes_done
        ds += [{'dist_to_start': dist_to_point, 'rte_id': rte} for rte in rtes_at_point]
        rtes_done = rtes_done.union(rtes_at_point)

    return pd.DataFrame(ds)


def bin_ride_distance(x, max_dist_from_start):
    if x < 1:
        return 0
    if x < max_dist_from_start / 5:
        return 0.5
    if x < max_dist_from_start / 2:
        return 3
    else:
        return 10
