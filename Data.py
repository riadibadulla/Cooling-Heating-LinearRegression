import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Data:
    x_values = []
    y_values = []
    x = None
    y = None
    fig = plt.figure()
    InputOutputVariables = []

    def __init__(self,x,y, InputOutputVariables):
        self.x = x
        self.y = y
        self.x_values = x.values
        self.y_values = y.values
        self.InputOutputVariables = InputOutputVariables

    def plotInputScatterMatrix(self):
        pd.plotting.scatter_matrix(self.x,alpha=0.5,marker='*')

    def plotData(self,x,y,subplotNumber):
        ax = self.fig.add_subplot(331+subplotNumber)
        ax.grid(color='lightgray', linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
        ax.scatter(x, y,color='red', alpha=.1, s=140, marker='.')
        ax.set_xlabel('x'+str(subplotNumber+1))
        ax.set_ylabel('y1')
        plt.savefig('plot_scatter2D.png')

    def plotAllInputOutput(self):
        for i in range(8):
            self.plotData(self.x_values[:,i],self.y_values[:,0],i)
        plt.show()