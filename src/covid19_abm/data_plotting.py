import matplotlib as mpl
import pylab as plt
import seaborn as sns
import pandas as pd
import json
import os
import geopandas as gpd

from covid19_abm.dir_manager import get_data_dir

sns.set(style="darkgrid")


def get_old_and_new_dist_shape():
    old_new_dist = pd.read_csv(
        get_data_dir('raw', 'district_relation.csv'))
    old_new_dist_map = old_new_dist[['DIST2012', 'NEW_DIST_ID_2']].set_index('DIST2012').to_dict()['NEW_DIST_ID_2']
    pop_dense = pd.read_csv(
        get_data_dir('raw', 'district_pop_dens_friction.csv'))
    old_new_dist = old_new_dist.merge(pop_dense, on='DIST2012')

    # p = gpd.read_file('../data/geo2_zw2012.shp')
    shape = p = gpd.read_file(
        get_data_dir('raw', 'shapefiles', 'new_districts', 'ZWE_adm2.shp'))
    shape['total_population'] = shape['ID_2'].map(old_new_dist.groupby('NEW_DIST_ID_2')['total_pop'].sum())

    return old_new_dist, shape


def load_model_logs_df(log_name):
    with open(log_name) as fl:
        content = fl.read()

    data = []
    prev_case = 0
    prev_hosp = 0
    prev_critical = 0

    for days, l in enumerate(content.strip().split('\n')):
        l = l.split(' $$ ')[-1]
        l = json.loads(l)

        total_infected = l['infected_count']
        total_hosp = l['hospitalized_count']
        total_critical = l['critical_count']

        current_case = total_infected - prev_case
        prev_case = total_infected

        current_hosp = total_hosp - prev_hosp
        prev_hosp = total_hosp

        current_critical = total_critical - prev_critical
        prev_critical = total_infected

        l['new_cases'] = current_case
        l['new_hospitalized'] = current_hosp
        l['new_critical'] = current_critical

        l['version'] = log_name
        l['days'] = days
        data.append(l)

    return pd.DataFrame(data)


def set_plot_config(df):
    plot_config = [
        dict(x='date', y='Deaths', hue='scenario', data=df, title='Cumulative deaths', y_label='Number of deaths', legend_loc='upper left'),
        dict(x='date', y='Infected', hue='scenario', data=df, title='Cumulative infected incidence', y_label='Number of infected', legend_loc='upper left'),
        dict(x='date', y='New cases', hue='scenario', data=df, title='Daily new cases', y_label='New cases', legend_loc='upper right'),
        dict(x='date', y='Active cases', hue='scenario', data=df, title='Daily active cases', y_label='Active cases', legend_loc='upper right'),
        dict(x='date', y='Hospitalizations', hue='scenario', data=df, title='Cumulative hospitalizations', y_label='Number of hospitalized', legend_loc='upper left'),
        dict(x='date', y='Current hospitalizations', hue='scenario', data=df, title='Current hospitalizations', y_label='Number of hospitalized', legend_loc='upper right'),
        dict(x='date', y='ICU admissions', hue='scenario', data=df, title='Cumulative ICU admissions', y_label='Number of admissions', legend_loc='upper left'),
        dict(x='date', y='Current ICU admissions', hue='scenario', data=df, title='Current ICU admissions', y_label='Number of admissions', legend_loc='upper right'),
    ]

    return plot_config


def plot_trajectories(x, y, hue, data, title, y_label, legend_loc, n_boot=100, in_thousands=True, figsize=(8, 4), dpi=300):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()

    if in_thousands:
        data = data.copy()
        data[y] = data[y] / 1000
        y_label = f'{y_label} (000s)'

    sns.lineplot(x=x, y=y, hue=hue,
                 data=data, ax=ax, n_boot=n_boot)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=6, title_fontsize=6, loc=legend_loc)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    ax.patch.set_alpha(0)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
