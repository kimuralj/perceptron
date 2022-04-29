import numpy as np
import csv
from perceptron import Perceptron


# Open data file
file_name = './dados/cars.csv'

# For multiclass problems:
# The class of the data has to be a number starting from 0
# Each different class receives a new number to represent it
# Example:
# X1    X2      Y   >>> convert to >>>> Class Number
# 0     0       A                       0
# 0     1       B                       1
# 1     0       C                       2
# 1     1       A                       0


# Initialize variables
X = []
Y = []

# Open the csv file and split X and Y
with open (file_name, mode='r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        row_size = np.size(row)         # Get the size of the row
        row = [float(x) for x in row]   # Convert each value in the row to a number
        X.append(row[:row_size-1])      # Append the N-1 items of the row to the X variable
        Y.append(row[row_size-1])       # Append the last item of the row to the Y variable

# Get the number of classes
N_class = np.size(np.unique(Y))

# If it's not just a binary problem
if N_class > 2:
    # Split Y in different classes
    for index, y in enumerate(Y):
        y_new = []
        for i in range(N_class):
            y_new.append(0 if i!=y else 1)
        Y[index] = y_new

Y = np.array(Y)
X = np.array(X)

network = Perceptron(eta=0.01, iteractions=10000, classes=N_class)

# Training perceptron
network.train(X,Y)

# Predicted result
predicted_value = network.classify(X)
print(np.round(np.array(predicted_value),2))
predicted_value = network.predict(X)
print(np.array(predicted_value))

# Test with new input
newX = np.array([[3,1,1]])

predicted_value = network.classify(newX)
print(np.round(np.array(predicted_value),2))
predicted_value = network.predict(newX)
print(np.array(predicted_value))