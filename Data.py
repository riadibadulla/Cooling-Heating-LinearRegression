import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Data:
    x_values = []                   #list of values for input
    y_values = []                   #list of values for output
    x = None                        #Input values in dataframe
    y = None                        #Output values in dataframe
    df = None                       #Full dataframe
    fig = None                      #matplotlib figure

    def __init__(self,df):
        self.df = df
        self.df = self.df.drop("X1",1)      #Drop X1 as it is correlated with X2
        self.df = self.df.drop("X4",1)      #Frop X4 as it is correlated with X5
        self.x = df.loc[: , 'X1':'X8']      #get input from data
        self.y = df.loc[: ,'Y1':'Y2']       #get output from data
        self.x_values = self.x.values       #array of values for input
        self.y_values = self.y.values       #array of values for output

    "Plots scatter matrix to see the correlation"
    def plotInputScatterMatrix(self):
        pd.plotting.scatter_matrix(self.x,alpha=0.2,marker='*')
        plt.show()

    "Plot sublpots"
    def plotData(self,x,y,subplotNumber,yNumber):
        ax = self.fig.add_subplot(331+subplotNumber)
        ax.grid(color='lightgray', linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
        ax.scatter(x, y,color='red', alpha=.1, s=140, marker='.')
        ax.set_xlabel('x'+str(subplotNumber+1))
        ax.set_ylabel(yNumber)
        plt.savefig('plot_scatter2D.png')

    "calls plotData and calls figure with all graphs for input given output"
    def plotAllInputOutput(self):
        for j in range(2):
            self.fig = plt.figure()
            for i in range(len(self.x_values[0])):
                self.plotData(self.x_values[:,i],self.y_values[:,j],i,"y"+str(j+1))
        plt.show()
    
    "Correlation"
    def correlation(self):
        import seaborn as sns
        corr = self.df.corr()
        sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True)
        plt.show()
        return corr

    def splitDataToTrainAdndTest(self):
        train, test = train_test_split(self.df, test_size=0.35)
        train_x = train.loc[: , 'X1':'X8']
        train_y = train.loc[: ,'Y1':'Y2']
        test_x = test.loc[: , 'X1':'X8']
        test_y = test.loc[: ,'Y1':'Y2']
        return (train_x, train_y, test_x, test_y)