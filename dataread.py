# Agent Numbrer : 180029410

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure()
def plotData(x,y,subplotNumber):
    ax = fig.add_subplot(331+subplotNumber)
    # Use a light grey grid to make reading easier, and put the grid 
    # behind the display markers
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    # Draw a scatter plot of the first column of x vs second column.
    ax.scatter(x, y,color='red', alpha=.1, s=140, marker='.')
    ax.set_xlabel('x'+str(subplotNumber+1))
    ax.set_ylabel('y1')

    # Save as an image file
    plt.savefig('plot_scatter2D.png')

    # Display in a window

# Load the data from the comma-separated csv file. 
df = pd.read_csv("ENB2012_data.csv")
x = df.loc[: , 'X1':'X8']
y = df.loc[: ,'Y1':'Y2']

x_values = x.values
y_values = y.values

print(x_values)
print(y_values)

# Create a new figure and an axes objects for the subplot
# We only have one plot here, but it's helpful to be consistent

for i in range(8):
    plotData(x_values[:,i],y_values[:,0],i)
plt.show()

