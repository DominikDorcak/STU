import pandas as pd
import numpy as np
from matplotlib import pyplot


data = pd.read_csv("data/players_20.csv")
data = data.drop(columns=data.columns[-27:-1])
data = data.drop(columns=data.columns[-1])
colstoremove = []
for col in data.columns:
    if col.startswith(('gk','goalkeeping')):
        colstoremove.append(col)

data = data.drop(columns=colstoremove)


print(len(data.columns))
logfile = open('numeric_data.txt','w+')


descr = data.describe()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
     print(descr, file=logfile)


logfile = open('categorical_data.txt','w+')
descr = data.describe(exclude=[np.number])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
     print(descr, file=logfile)


positioncounts = data.groupby('player_positions').size()
print(positioncounts)
datanumeric = data.select_dtypes([np.number])
print(datanumeric.skew())


def draw_histograms(data_to_plot):
    for col in data_to_plot.columns:
        print('vykreslujem histogram pre ' + col)
        figure = pyplot.figure()
        ax = figure.add_subplot(111)
        ax.set_xlabel(col)
        ax.set_ylabel('pocet')
        data_to_plot[col].hist()
        figure.savefig('plot/hist/'+col+'.png')
        pyplot.close(figure)


def draw_density(data_to_plot):
    for col in data_to_plot.columns:
        print('vykreslujem density pre ' + col)
        figure = pyplot.figure()
        ax = figure.add_subplot(111)
        ax.set_xlabel(col)
        ax.set_ylabel('pocet')
        data_to_plot[col].plot(kind='density')
        figure.savefig('plot/density/'+col+'.png')
        pyplot.close(figure)


def draw_boxes(data_to_plot):
    for col in data_to_plot.columns:
        print('vykreslujem box plot pre ' + col)
        figure = pyplot.figure()
        data_to_plot[col].plot(kind='box')
        figure.savefig('plot/box/'+col+'.png')
        pyplot.close(figure)


def draw_corr(data_to_plot,method):
    correlations = data_to_plot.corr(method)
    fig = pyplot.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations,vmin=-1,vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,50,1)
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(data_to_plot.columns,rotation='vertical')
    ax.set_yticklabels(data_to_plot.columns)
    fig.savefig('plot/corr_'+method+'.png')
    pyplot.close(fig)


draw_density(datanumeric)
draw_histograms(datanumeric)
draw_boxes(datanumeric)
draw_corr(datanumeric,'pearson')
draw_corr(datanumeric,'kendall')
draw_corr(datanumeric,'spearman')