# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:45:40 2022

@author: ASUS
"""
#IMPORT LIBRARY
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import quandl
import pickle
import plotly_express as px
from plotly.offline import plot
import visualkeras

#LOAD DATA
id="1-u_uaNVF6XmsfYZxtWR"  
def get_quandl_data(id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(id))
        df = quandl.get(id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(id, cache_path))
    return df

# Pull Kraken BTC price exchange data
btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')

#EDA
l=['Open','High','Low','Close','Volume (BTC)','Volume (Currency)','Weighted Price']
for i in l:
    print(i)
    btc_trace = go.Scatter(x=btc_usd_price_kraken.index, y=btc_usd_price_kraken[i])
    py.iplot([btc_trace])
#SCALING
df_for_training = btc_usd_price_kraken[l].astype(float)
#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)
    

#DATA DEFINING(SHAPE)
#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 7. We will make timesteps = 10 (past days data used for training). 

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 10  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (2713, 7)
#2713 refers to the number of data points and 7 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future,0:])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define the Autoencoder LSTM model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

plt.figure(figsize=(15,15))
#visualkeras.layered_view(model).show() # display using your system viewer
#visualkeras.layered_view(model, to_file='LSTM_MODEL_ARCH.png') # write to disk
from keras_sequential_ascii import keras2ascii
keras2ascii(model)

#from eiffel2 import builder

# or the following if you want to have a dark theme
#builder([7,64,32,32,1], bmode="night")
model.save("my_h5_lstmmodel.h5")

# fit the model
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
pd.DataFrame(history.history)

#Predicting...
#Libraries that will help us extract only business days in the US.
#Otherwise our dates would be wrong when we look back (or forward).  
#from pandas.tseries.holiday import USFederalHolidayCalendar
#rom pandas.tseries.offsets import CustomBusinessDay
#us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
#Remember that we can only predict one day in future as our model needs 7 variables
#as inputs for prediction. We only have all 7 variables until the last day in our dataset.
#n_past = 16
#n_days_for_prediction=15  #let us predict past 15 days
train_dates=btc_usd_price_kraken.index


#NO OF DAYS FORCASTING
n_future=30 
#starting from last date of train dates
forcast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()
print(forcast_period_dates)

#Make prediction
prediction = model.predict(trainX[-n_future:]) #shape = (n, 1) where n is the n_days_for_prediction

#Perform inverse transformation to rescale back to original range
#Since we used 7 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 7 times and discard them after inverse transform
forcast_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forcast_copies)[:]

# Convert timestamp to date
forecast_dates = []
for time_i in forcast_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future[:,0]})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


org= pd.DataFrame({'Date':train_dates, 'Open':btc_usd_price_kraken['Open']})
org['Date']=pd.to_datetime(org['Date'])
org = org.loc[org['Date'] >= '2021-5-25']
org.shape

a=sns.lineplot(org['Date'], org['Open'])
sns.lineplot(df_forecast['Date'][5:,], df_forecast['Open'],)


