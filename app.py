# -*- coding: utf-8 -*-

import io
import random
import datetime

from flask import Flask, Response, make_response, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from matplotlib import collections, colors, transforms

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')


app = Flask(__name__)

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("darkgrid")

# Import dataset
mpg_df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")

def plot_fig(figure):
    fig = figure
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simple.plot')
def simple_plot():
    fig = create_figure()
    return plot_fig(fig)

def create_figure():
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    ax.plot(xs, ys)
    return fig


@app.route("/plotdate.plot")
def plotdate_plot():
    fig = Figure()
    ax = fig.add_subplot(111)
    x = []
    y = []
    now = datetime.datetime.now()
    delta = datetime.timedelta(days = 1)
    for i in range(10):
        x.append(now)
        now += delta
        y.append(random.randint(0, 1000))
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route("/scatter.plot")
def scatter_plot():
    fig = create_scatter_figure()
    return plot_fig(fig)

def create_scatter_figure():
    # Prepare Data
    # Create as many colors as there are unique midwest['category']
    categories = np.unique(midwest['category'])
    colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

    fig = Figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
    axes = fig.add_subplot(111, xlim=(0.0, 0.1), ylim=(0, 90000),
                xlabel=u'面积', ylabel='Population',
                title="Scatterplot of Midwest Area vs Population")

    for i, category in enumerate(categories):
        axes.scatter('area', 'poptotal',
                    data=midwest.loc[midwest.category==category, :],
                    s=20, cmap=colors[i], label=str(category))
                    # "c=" 修改为 "cmap="，Python数据之道 备注
    return fig


@app.route("/jittering.plot")
def jittering_plot():
    fig = create_jitterting_figure()
    return plot_fig(fig)

def create_jitterting_figure():
    # Import Data

    # Draw Stripplot
    fig = Figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
    axes = fig.add_subplot(111)
    sns.stripplot(mpg_df.cty, mpg_df.hwy, jitter=0.25, size=8, ax=axes, linewidth=.5)
    return fig


@app.route("/counts.plot")
def counts_plot():
    fig = create_counts_figure()
    return plot_fig(fig)

def create_counts_figure():
    # Import Data
    df_counts = mpg_df.groupby(['hwy', 'cty']).size().reset_index(name='counts')

    # Draw Stripplot
    fig = Figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
    axes = fig.add_subplot(111, title="Counts Plot - Size of circle is bigger as more points overlap")
    sns.stripplot(df_counts.cty, df_counts.hwy, size=df_counts.counts*2, ax=axes)
    return fig


@app.route("/densitycurve.plot")
def desity_curve_plot():
    fig = create_desity_curve_figure()
    return plot_fig(fig)

def create_desity_curve_figure():
    # Plot
    fig = Figure(figsize=(16, 10), dpi=80)
    axes = fig.add_subplot(111, ylim=(0, 0.35),
                        title="Density Plot of City Mileage by Vehicle Type")
    sns.distplot(mpg_df.loc[mpg_df['class'] == 'compact', "cty"], color="dodgerblue",
                label="Compact", hist_kws={'alpha':.7}, kde_kws={'linewidth':3},
                ax=axes)
    sns.distplot(mpg_df.loc[mpg_df['class'] == 'suv', "cty"], color="orange",
                label="SUV", hist_kws={'alpha':.7}, kde_kws={'linewidth':3},
                ax=axes)
    sns.distplot(mpg_df.loc[mpg_df['class'] == 'minivan', "cty"], color="g",
                label="minivan", hist_kws={'alpha':.7}, kde_kws={'linewidth':3},
                ax=axes)

    # Decoration
    # fig.legend()
    return fig


@app.route("/distribution.plot")
def distribution_plot():
    fig = create_distribution_figure()
    return plot_fig(fig)

def create_distribution_figure():
    rs = np.random.RandomState(10)

    # Set up the matplotlib figure
    fig = Figure(figsize=(14, 14), dpi=80)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, sharex=ax1)
    ax3 = fig.add_subplot(223, sharex=ax1)
    ax4 = fig.add_subplot(224, sharex=ax1)

    # Generate a random univariate dataset
    d = rs.normal(size=100)

    # Plot a simple histogram with binsize determined automatically
    sns.distplot(d, kde=False, color="b", ax=ax1)

    # Plot a kernel density estimate and rug plot
    sns.distplot(d, hist=False, rug=True, color="r", ax=ax2)

    # Plot a filled kernel density estimate
    sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=ax3)

    # Plot a historgram and kernel density estimate
    sns.distplot(d, color="m", ax=ax4)
    return fig


@app.route("/kde.plot")
def kde_plot():
    fig = create_kde_figure()
    return plot_fig(fig)

def create_kde_figure():
    # Draw Plot
    fig = Figure(figsize=(13, 10), dpi=80)
    axes = fig.add_subplot(111, ylim=(0, 0.35),
                        title='Density Plot of City Mileage by n_Cylinders')
    sns.kdeplot(mpg_df.loc[mpg_df['cyl'] == 4, "cty"], shade=True, color="g", 
                label="Cyl=4", alpha=.7, ax=axes)
    sns.kdeplot(mpg_df.loc[mpg_df['cyl'] == 5, "cty"], shade=True, color="deeppink", 
                label="Cyl=5", alpha=.7,ax=axes)
    sns.kdeplot(mpg_df.loc[mpg_df['cyl'] == 6, "cty"], shade=True, color="dodgerblue", 
                label="Cyl=6", alpha=.7, ax=axes)
    sns.kdeplot(mpg_df.loc[mpg_df['cyl'] == 8, "cty"], shade=True, color="orange", 
                label="Cyl=8", alpha=.7, ax=axes)
    # Decoration
    fig.legend()
    return fig


@app.route("/box.plot")
def box_plot():
    fig = create_box_figure()
    return plot_fig(fig)

def create_box_figure():
    sns.set_style("white")
    # Draw Plot
    fig = Figure(figsize=(13, 10), dpi=80)
    axes = fig.add_subplot(111, ylim=(10, 40),
                        title="Box Plot of Highway Mileage by Vehicle Class")
    sns.boxplot(x='class', y='hwy', data=mpg_df, notch=False, ax=axes)
    for i in range(len(mpg_df['class'].unique())-1):
        axes.vlines(i+.5, 10, 45, linestyles='solid', colors='gray', alpha=0.2)
    return fig

@app.route("/dotbox.plot")
def dotbox_plot():
    fig = create_dotbox_figure()
    return plot_fig(fig)

def create_dotbox_figure():
    sns.set_style("white")
    # Draw Plot
    fig = Figure(figsize=(13, 10), dpi=80)
    axes = fig.add_subplot(111, title="Box Plot of Highway Mileage by Vehicle Class")


    sns.boxplot(x='class', y='hwy', data=mpg_df, hue='cyl', ax=axes)
    sns.stripplot(x='class', y='hwy', data=mpg_df, 
                color='black', size=3, jitter=1, ax=axes)

    for i in range(len(mpg_df['class'].unique())-1):
        axes.vlines(i+.5, 10, 45, linestyles='solid', colors='gray', alpha=0.2)

    # Decoration
    fig.legend(title='Cylinders')
    return fig

@app.route("/violion.plot")
def violion_plot():
    fig = create_violion_figure()
    return plot_fig(fig)

def create_violion_figure():
    # Draw Plot
    fig = Figure(figsize=(13, 10), dpi=80)
    axes = fig.add_subplot(111, title="Violin Plot of Highway Mileage by Vehicle Class")

    sns.violinplot(x='class', y='hwy', data=mpg_df, scale='width',
                inner='quartile', ax=axes)
    return fig