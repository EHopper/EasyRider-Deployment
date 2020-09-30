import json
import collections
import pandas as pd
import numpy as np

from util import config


def load_gridpts(df_filename, dict_filename):
    grid_pts = pd.read_csv(config.MODEL_PATH + df_filename + '.csv')
    with open(config.MODEL_PATH + dict_filename + '.json', 'r') as fn:
        gd = json.load(fn)
    grid_dict = dict()
    for key in gd:
        grid_dict[int(key)] = set(gd[key])

    return grid_pts, grid_dict

def set_presets():
    df = pd.read_csv(config.PROCESSED_DATA_PATH + 'trips.csv')

    presets_descriptions = [
        'Chilling out in the saddle', 'Pretty relaxed, with some climbing',
        'Half-day of touring', 'Training for VO2-max', 'Training for strength',
        'Training for a century']

    presets = pd.DataFrame({
        'distance': [10., 15., 45., 20., 10., 85.],
        'avg_slope': [2., 5., 3., 2., 8., 2.],
        'avg_speed': [10., 8., 15., 18., 8., 15.],
        'prop_moving': [0.7, 0.7, 0.8, 1., 0.7, 0.8]
    })

    return presets, presets_descriptions

def remove_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_csv(config.MODEL_PATH + 'feature_scaling.csv', index_col=0)
    df_unscale = df.copy()
    for col in scaler.columns:
        df_unscale[col] = (df_unscale[col] / scaler.loc['column_importance', col]
                    * scaler.loc['std', col] + scaler.loc['mean', col])

    return df_unscale

def apply_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_csv(config.MODEL_PATH + 'feature_scaling.csv', index_col=0)
    df_scale = df.copy()
    for col in scaler.columns:
        df_scale[col] = ((df_scale[col] - scaler.loc['mean', col])
                            / scaler.loc['std', col]
                            * scaler.loc['column_importance', col])

    return df_scale