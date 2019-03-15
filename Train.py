from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
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

    def plotData(self,x,y,predictedY, subplotNumber,yNumber,title):
        self.fig.suptitle(title)
        ax = self.fig.add_subplot(331+subplotNumber)
        ax.grid(color='lightgray', linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
        real = ax.scatter(x, y,color='red', alpha=.6, s=140, marker='.')
        pred = ax.scatter(x, predictedY,color='blue', alpha=.7, s=140, marker='.')
        self.fig.legend([real,pred],["real values","predicted values"])
        ax.set_xlabel('x'+str(subplotNumber+1))
        ax.set_ylabel(yNumber)
        plt.savefig('plot_scatter2D.png')

    "Training data plot"
    def plotAllInputOutput(self,predictedY):
        for j in range(2):
            self.fig = plt.figure()
            for i in range(len(self.train_x.values[0])):
                self.plotData(self.train_x.values[:,i],self.train_y.values[:,j],predictedY[:,j], i,"y"+str(j+1), "Training values")
        plt.show()
    
    "Testing data plot"
    def plotAllInputOutputTest(self,predictedY):
        for j in range(2):
            self.fig = plt.figure()
            for i in range(len(self.train_x.values[0])):
                self.plotData(self.test_x.values[:,i], self.test_y.values[:,j], predictedY[:,j], i,"y"+str(j+1), "Testing values")
        plt.show()

    def trainViaLinearRegression(self,plot):
        print("\n\n\nLinear regression")
        linreg = LinearRegression(normalize=True)
        x = self.train_x.values
        y = self.train_y.values
        linreg.fit(x,y)
        y_hat = linreg.predict(self.test_x.values)
        y_hat_train = linreg.predict(x)
        print('MSE = ', mean_squared_error(self.test_y.values,y_hat))
        #print(r2_score(self.test_y.values,y_hat))
        if plot:
            self.plotAllInputOutput(linreg.predict(self.train_x))
            self.plotAllInputOutputTest(linreg.predict(self.test_x))
        return (self.train_x, linreg.predict(self.train_x))

    def findTheBestPolynomialDegree(self):
        print("\n\n\nBEST POLYNOMIAL DEGREE")
        x = self.train_x.values
        y = self.train_y.values
        MSE_min = 2
        bestOrder = 0
        for i in range(1,10):
            poly = PolynomialFeatures(degree = i)
            X_poly = poly.fit_transform(x)
            X_poly_test = poly.fit_transform(self.test_x.values)
            lin_reg = LinearRegression()
            lin_reg.fit(X_poly, self.train_y)
            y_hat = lin_reg.predict(X_poly_test)
            MSE = mean_squared_error(self.test_y.values,y_hat)
            print("Order- ",i," MSE = ",MSE)
            if (MSE < MSE_min):
                MSE_min = MSE
                bestOrder = i
        print("Best Order- ",bestOrder," MSE = ",MSE_min)
    
    def polynomial(self, plot):
        print("\n\n\nPOLYNOMIAL Regression")
        x = self.train_x.values
        y = self.train_y.values
        poly = PolynomialFeatures(degree = 4)
        X_poly = poly.fit_transform(x)
        X_poly_test = poly.fit_transform(self.test_x.values)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, self.train_y)
        y_hat = lin_reg.predict(X_poly_test)
        print('MSE (Polynomial 4) = ', mean_squared_error(self.test_y.values,y_hat))
        if plot:
            self.plotAllInputOutput(lin_reg.predict(X_poly))
            self.plotAllInputOutputTest(lin_reg.predict(X_poly_test))

    def kFold(self):
        print("\n\n\n\nCross validation")
        train_x = self.train_x.values
        train_y = self.train_y.values
        poly = PolynomialFeatures(degree = 4)

        kf = KFold(n_splits=50)
        kf.get_n_splits(train_x)
        print(kf)  
        lin_reg = LinearRegression()
        for train_index, validation_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_validation = train_x[train_index], train_x[validation_index]
            y_train, y_validation = train_y[train_index], train_y[validation_index]
            X_train_poly = poly.fit_transform(X_train)
            x_validation_poly = poly.fit_transform(X_validation)
            lin_reg.fit(X_train_poly, y_train)
            print("MSE K Fold = ", mean_squared_error(lin_reg.predict(x_validation_poly),y_validation))
        
        
        