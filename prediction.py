# HDFC-Stock-Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('NSE-HDFCBANK.csv')
X = dataset.iloc[:,[1,2,3,4,5,6]].values
y = dataset.iloc[:, 7].values
dates=dataset.iloc[:,0:1].values
open_y=dataset.iloc[:,1].values
high_y=dataset.iloc[:,2].values
low_y=dataset.iloc[:,3].values
last_y=dataset.iloc[:,4].values
close_y=dataset.iloc[:,5].values
ttq_y=dataset.iloc[:,6].values

# Encoding dates(catetgorical variables)
from sklearn.preprocessing import LabelEncoder
labelencoder_dates=LabelEncoder()
dates=labelencoder_dates.fit_transform(dates)
dates=np.reshape(dates, (4947,1))

# Fitting the regressors to the individual datasets
from sklearn.tree import DecisionTreeRegressor
regressor_open=DecisionTreeRegressor(random_state=0)
regressor_open.fit(dates,open_y)
regressor_high=DecisionTreeRegressor(random_state=0)
regressor_high.fit(dates,high_y)
regressor_low=DecisionTreeRegressor(random_state=0)
regressor_low.fit(dates,low_y)
regressor_last=DecisionTreeRegressor(random_state=0)
regressor_last.fit(dates,last_y)
regressor_close=DecisionTreeRegressor(random_state=0)
regressor_close.fit(dates,close_y)
regressor_ttq=DecisionTreeRegressor(random_state=0)
regressor_ttq.fit(dates,ttq_y)

# Plot of "Open"
X_grid=np.arange(min(dates), max(dates), 0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(dates, open_y, color='red')
plt.plot(X_grid, regressor_open.predict(X_grid), color='blue')
plt.title('HDFC Bank(Open)')
plt.xlabel('Date')
plt.ylabel('Open')
plt.show()

# Plot of "High"
X_grid=np.arange(min(dates), max(dates), 0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(dates, high_y, color='red')
plt.plot(X_grid, regressor_high.predict(X_grid), color='blue')
plt.title('HDFC Bank(High)')
plt.xlabel('Date')
plt.ylabel('High')
plt.show()

# Plot of "Last"
X_grid=np.arange(min(dates), max(dates), 0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(dates, last_y, color='red')
plt.plot(X_grid, regressor_last.predict(X_grid), color='blue')
plt.title('HDFC Bank(Last)')
plt.xlabel('Date')
plt.ylabel('Last')
plt.show()

# Plot of "Close"
X_grid=np.arange(min(dates), max(dates), 0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(dates, close_y, color='red')
plt.plot(X_grid, regressor_close.predict(X_grid), color='blue')
plt.title('HDFC Bank(Close)')
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()


# Plot of "Total Trade Quantity(TTQ)"
X_grid=np.arange(min(dates), max(dates), 0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(dates, ttq_y, color='red')
plt.plot(X_grid, regressor_ttq.predict(X_grid), color='blue')
plt.title('HDFC Bank(Total Trade Quantity)')
plt.xlabel('Date')
plt.ylabel('TTQ')
plt.show()



# Predicting the results
pred_open = regressor_open.predict(4947)
pred_high=regressor_high.predict(4947)
pred_low=regressor_low.predict(4947)
pred_last=regressor_last.predict(4947)
pred_close=regressor_close.predict(4947)
pred_ttq=regressor_ttq.predict(4947)

# Combining all the predicted results into a csv file and importing it
predicted_data=pd.read_csv('Predicted_Data.csv')
predicted_data_x=predicted_data.iloc[:,1:7]

# Fitting and predicted the final result(turnover)
from sklearn.ensemble import RandomForestRegressor
regressor_final=RandomForestRegressor(n_estimators=14, random_state=0)
regressor_final.fit(X, y)
predicted_turnover=regressor_final.predict(predicted_data_x)












