import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import ShuffleSplit,cross_val_score,KFold,LeaveOneOut
from sklearn.linear_model import LogisticRegression
seed=42


data = pd.read_csv("data/players_20.csv")
data = data.drop(columns=data.columns[-27:-1])
data = data.drop(columns=data.columns[-1])
colstoremove = ['sofifa_id','player_url','dob']
indexestoremove = []
for d in range(len(data.index)):
    if not np.isnan(data['gk_reflexes'].values[d]):
        indexestoremove.append(d)

data = data.drop(indexestoremove)


for col in data.columns:
    if col.startswith(('gk','goalkeeping')):
        colstoremove.append(col)

data = data.drop(columns=colstoremove)


print(len(data.columns))
logfile = open('numeric_data.txt','w+')


descr = data.describe()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(descr, file=logfile)
    logfile.close()


logfile = open('categorical_data.txt','w+')
descr = data.describe(exclude=[np.number])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(descr, file=logfile)
    logfile.close()


positioncounts = data.groupby('team_position').size()
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


def standardize(dataframe):
    scaler = StandardScaler()
    rescaleddataframe = scaler.fit_transform(dataframe)
    return rescaleddataframe

columns = datanumeric.columns



# columns_to_train = []
# columns_to_train.extend(columns[0:9])
# columns_to_train.extend(columns[10:])
columns_to_train = [columns[3]]
columns_to_train.extend(columns[14:])
print(columns_to_train)
columns_to_test = [columns[9]]


data_to_train = datanumeric[columns_to_train]
data_to_test = datanumeric[columns_to_test]
print(datanumeric.dtypes)

rescaled = np.array(standardize(data_to_train)).astype('float64')
np.set_printoptions(precision=3)
rdf = pd.DataFrame(rescaled)
rdf.dropna()

data_to_train.dropna()
test = SelectKBest(score_func=chi2,k=15)
fit = test.fit(data_to_train,data_to_test)
finalcolumns = data_to_train.columns[test.get_support()]

print(finalcolumns)

features = fit.transform(data_to_train)
print(features.shape)

# draw_density(datanumeric)
# draw_histograms(datanumeric)
# draw_boxes(datanumeric)
# draw_corr(datanumeric,'pearson')
# draw_corr(datanumeric,'kendall')
# draw_corr(datanumeric,'spearman')


rescaled = standardize(np.array(data_to_train))
np.set_printoptions(precision=3)
print(rescaled.shape)

reduced_dataframe = pd.DataFrame(features,columns=finalcolumns)
classifier = ExtraTreesClassifier(n_estimators=200)
classifier.fit(features,np.array(data_to_test).flatten())
print(classifier.feature_importances_)

# kfold = ShuffleSplit(15,0.4,seed)
# kfold = KFold(15,True,seed)
kfold = LeaveOneOut()
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model,features,np.array(data_to_test).flatten(),cv=kfold)
print('Presnost: %.3f%% (%.3f%%)' % (results.mean()*100.0,results.std()*100.0))





fig = pyplot.figure()
correlations = pd.DataFrame(features).corr('pearson')
ax = fig.add_subplot(111)
cax = ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,15,1)
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_xticklabels(finalcolumns,rotation='vertical')
ax.set_yticklabels(finalcolumns)
pyplot.show()