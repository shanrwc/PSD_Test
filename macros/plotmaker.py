#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas.io.data
import matplotlib.pyplot as plt

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

##########################################################################
##                         Code Goes Here                               ##
##########################################################################

#Read in data from file
df = pd.read_csv('../data/hour.csv')

#Remove columns I don't want plots of
names = df.columns.values.tolist()
names.remove("instant")
names.remove("cnt")
names.remove("dteday")

#Make ga-bunches of scatter plots!
for n in names:
    makeScatterPlot(df[[n]],n,df[['cnt']],"Count",otherlabel="_corrcheck")

#Any other quantities seem interesing?  Make another scatter plot
makeScatterPlot(df.registered+df.casual,"Reg+Cas",df[['cnt']],"Count",otherlabel="_corrcheck")

#If you want to keep those new quantities, save a new csv file

