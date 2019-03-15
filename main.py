import Data
import pandas as pd
import Train
import os
os.system('clear')

InputOutputVariables = ["Relative Compactness", "Surface Area", "Wall Area" ,
                        "Roof Area" ,"Overall Height","Orientation" ,"Glazing area",
                        "Glazing area distribution","Heating Load","Cooling Load"]
data =None

def loadData():
    global data
    global InputOutputVariables
    df = pd.read_csv("ENB2012_data.csv")
    data = Data.Data(df)

def trainData():
    preparedData = data.splitDataToTrainAdndTest()
    model = Train.TrainingModel(preparedData[0],preparedData[1],preparedData[2],preparedData[3])
    model.trainViaLinearRegression(True)
    model.polynomial(True)
    model.findTheBestPolynomialDegree()
    model.kFold()

loadData()
data.plotInputScatterMatrix()
data.plotAllInputOutput()
trainData()