#!/usr/bin/python

import os
import sys
import itertools
import collections
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas.io.data
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

##########################################################################
##Helper Functions
def makeScatterPlot(xs,xlabel,ys,ylabel,otherlabel = "",m=None,b=None):
    plt.scatter(xs,ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if m is not None and b is not None:
        plt.plot(xs,m*xs+b,"-")
    plotname = xlabel+"_v_"+ylabel
    if len(otherlabel) > 0: plotname += otherlabel
    plt.savefig(plotname+".eps",format='eps',dpi=1000)
    plt.clf()

def makeHistogram(zs,zlabel,bins,finlabel=""):
    plt.hist(zs,bins,histtype='step',color='r',log=True)
    plt.xlabel(zlabel)
    plt.ylabel("Frequency")
    plt.ylim(0.9,15000)
    pname = zlabel + finlabel
    plt.savefig(pname+".eps",format='eps',dpi=1000)
    plt.clf()

##########################################################################
##                         Code Goes Here                               ##
##########################################################################

#Read in data and split off evaluation sample
df = pd.read_csv('../data/hour.csv')
np.random.seed(42)
rows = np.random.choice(df.index.values,0.1*len(df.index),replace=False)
evl_df = df.ix[rows]
mod_df = df.drop(rows)

##In outermost loop, pick out data for this interation
vlist = ["temp","weathersit","weekday","windspeed","atemp","hr","hum","workingday","yr","holiday","mnth","weekday","season"]
variables = [6,9,12]

min_max_scaler = MinMaxScaler()

outfile = open('regressor_score.txt','w')

for vcount in variables:
    features = mod_df[vlist]
    sfeatures = min_max_scaler.fit_transform(features)
    targets = mod_df[['cnt']]

    efeatures = evl_df[vlist]
    sefeatures = min_max_scaler.fit_transform(efeatures)
    etargets = evl_df[['cnt']]
    
    #Select vcount best features
    best = SelectKBest(f_regression,k=vcount)
    sfeatures = best.fit_transform(sfeatures,targets.values.ravel())
    sefeatures = best.transform(sefeatures)

    
    outfile.write("Selected Variables\n")
    finalvs = []
    for n,i in zip(vlist,best.get_support()): 
        if i: finalvs.append(n)
    outfile.write(" ".join(finalvs)+"\n\n")

    #Lasso Regression
    las = GridSearchCV(Lasso(),param_grid={'alpha':[0.5,0.75,1.0]},cv=KFold(len(sfeatures),5),scoring='mean_absolute_error')
    las.fit(sfeatures,targets.values.ravel())
    outfile.write("Lasso Regression\n")
    outfile.write("kFold Scores: "+str(las.grid_scores_)+"\n")
    outfile.write("Best Score: "+str(las.best_score_)+"\n")
    ws = las.best_estimator_.coef_
    outfile.write("Variable coefficients: \n")
    for n,w in zip(vlist,ws):
        outfile.write(str(n)+"   "+str(w)+"\n")
    outfile.write("Global Model Score: "+str(las.score(sfeatures,targets))+"\n")
    outfile.write("Evaluation Score: "+str(las.score(sefeatures,etargets))+"\n\n\n")
    errors = etargets.values.ravel()-las.predict(sefeatures)
    makeHistogram(errors,"Lasso_Errors",100,finlabel="_"+str(vcount))

    #Random Forest
    params={'n_estimators':[10,50,100],'max_depth':[4,8,12],'min_samples_split':[2,6,10,14]}
    forest = RandomForestRegressor(random_state=42)
    trees = GridSearchCV(forest,param_grid=params,cv=KFold(len(sfeatures),5),scoring='mean_absolute_error')
    trees.fit(sfeatures,targets.values.ravel())
    outfile.write("Random Forest\n")
    outfile.write("kFold Scores: "+str(trees.grid_scores_)+"\n")
    outfile.write("Best Score: "+str(trees.best_score_)+"\n")
    imps = trees.best_estimator_.feature_importances_
    for n,i in zip(vlist, imps):
        outfile.write(str(n)+"   "+str(i)+"\n")
    outfile.write("Global Model Score: "+str(trees.score(sfeatures,targets))+"\n")
    outfile.write("Evaluation Score: "+str(trees.score(sefeatures,etargets))+"\n\n\n")
    errors = etargets.values.ravel()-trees.predict(sefeatures)
    makeHistogram(errors,"Forest_Errors",100,finlabel="_"+str(vcount))
    outfile.write("--------------------------------------------------------------\n")

outfile.close()
