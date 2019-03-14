import Data
import pandas as pd

InputOutputVariables = ["Relative Compactness", "Surface Area", "Wall Area" ,
                        "Roof Area" ,"Overall Height","Orientation" ,"Glazing area",
                        "Glazing area distribution","Heating Load","Cooling Load"]
data =None

def loadData():
    global data
    global InputOutputVariables

    df = pd.read_csv("ENB2012_data.csv")
    x = df.loc[: , 'X1':'X8']
    y = df.loc[: ,'Y1':'Y2']
    data = Data.Data(x,y, InputOutputVariables)

loadData()
data.plotInputScatterMatrix()
data.plotAllInputOutput()

