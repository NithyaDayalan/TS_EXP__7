## Ex.No: 07 AUTO REGRESSIVE MODEL
### Date: 15/04/2025

### AIM:
To Implementat an Auto Regressive Model using Python

### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
   
### PROGRAM:
```
Devloped by: NITHYA D
Register Number: 2122232401110
```
#### Import necessary libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```
#### Load the dataset
```
data = pd.read_csv('Gold Price Prediction.csv')
```
#### Convert 'Date' to datetime and set as index
```
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.set_index('Date', inplace=True)
```
#### Choose a valid numeric column for prediction (e.g., 'Price Today')
```
series = data['Price Today'].dropna()  
```
#### Reduce to first 100 data points
```
series = series.iloc[:100]
```
#### Perform ADF test
```
result = adfuller(series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
#### Split into training and testing sets
```
x = int(0.8 * len(series))
train_data = series.iloc[:x]
test_data = series.iloc[x:]
```
#### Fit AR model
```
lag_order = 13
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()
```
#### Plot ACF and PACF
```
plt.figure(figsize=(10, 6))
plot_acf(series, lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Gold Price')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(series, lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Gold Price')
plt.show()
```
#### Make predictions
```
start = len(train_data)
end = len(train_data) + len(test_data) - 1
predictions = model_fit.predict(start=start, end=end)
```
#### Evaluate predictions
```
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)
```
#### Plot test data vs predictions
```
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Actual Gold Prices')
plt.plot(test_data.index, predictions, label='Predicted Gold Prices', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('AR Model Predictions vs Actual Gold Prices')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:

#### Result Data :
![image](https://github.com/user-attachments/assets/37ff10ab-aaa6-455a-b09a-da3aba503961)

#### ACF Plot :
![image](https://github.com/user-attachments/assets/b1d1ef37-4b85-4476-9758-4907c1ee554d)
#### PACF Plot :
![image](https://github.com/user-attachments/assets/5bf2e08b-93d6-4f53-a2e6-32301e6f4a35)

#### Prediction vs test data :
![image](https://github.com/user-attachments/assets/dedf41a3-29b7-4457-9344-e7646ca1b159)

#### Accuracy :
![image](https://github.com/user-attachments/assets/82e5e16b-8028-414a-9885-3f6d1e66abe4)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
