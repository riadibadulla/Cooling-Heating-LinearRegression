from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

    def trainViaLinearRegression(self):
        linreg = LinearRegression()
        linreg.fit(self.train_x,self.train_y)
        y_hat = linreg.predict(self.test_x)
        print('MSE = ', mean_squared_error(self.test_y,y_hat))
        return (self.train_x, linreg.predict(self.train_x))