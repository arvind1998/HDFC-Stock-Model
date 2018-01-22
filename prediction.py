# HDFC-Stock-Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('NSE-HDFCLIFE.csv')
X = dataset.iloc[:,[1,2,3,4,5,6]].values
y = dataset.iloc[:, 7].values
dates=dataset.iloc[:,0].values
today=pd.read_csv('TODAY.csv')
today_X = today.iloc[:,[1,2,3,4,5,6]].values
today_Y = today.iloc[:, 7].values

# Encoding dates(catetgorical variables)
from sklearn.preprocessing import LabelEncoder
labelencoder_dates=LabelEncoder()
dates=labelencoder_dates.fit_transform(dates)

# Creating the Regressor and fitting it to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=14, random_state=0)
regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict(today_X)

# Plotting the graph
plt.scatter(dates, y, color = 'red')
plt.plot(dates, y, color = 'blue')
plt.title('HDFC Stocks')
plt.xlabel('Dates')
plt.ylabel('Turnover(in Lakhs)')
plt.show()








