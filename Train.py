from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
class TrainingModel:

    train_x = None
    train_y = None
    test_x = None
    test_y = None

    def __init__(self,train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def plotData(self,x,y,predictedY, subplotNumber,yNumber):
        ax = self.fig.add_subplot(331+subplotNumber)
        ax.grid(color='lightgray', linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
        ax.scatter(x, y,color='red', alpha=.4, s=140, marker='.')
        ax.plot(x, predictedY,color='blue')
        ax.set_xlabel('x'+str(subplotNumber+1))
        ax.set_ylabel(yNumber)
        plt.savefig('plot_scatter2D.png')

    def plotAllInputOutput(self,predictedY):
        for j in range(2):
            self.fig = plt.figure()
            for i in range(8):
                self.plotData(self.train_x[:,i],self.train_y[:,j],predictedY[:,j], i,"y"+str(j+1))
        plt.show()
    
    def plotAllInputOutputTest(self,predictedY):
        for j in range(2):
            self.fig = plt.figure()
            for i in range(8):
                self.plotData(self.test_x[:,i], self.test_y[:,j], predictedY[:,j], i,"y"+str(j+1))
        plt.show()

    def trainViaLinearRegression(self,plot):
        linreg = LinearRegression(normalize=True)
        x = self.train_x.values
        y = self.train_y.values
        linreg.fit(x,y)
        y_hat = linreg.predict(self.test_x.values)
        y_hat_train = linreg.predict(x)
        print('MSE Train = ', mean_squared_error(y,y_hat_train))
        print('MSE = ', mean_squared_error(self.test_y.values,y_hat))
        print(r2_score(self.test_y.values,y_hat))
        if plot:
            self.plotAllInputOutput(linreg.predict(self.train_x))
            self.plotAllInputOutputTest(linreg.predict(self.test_x))
        return (self.train_x, linreg.predict(self.train_x))