import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data #to scrap
import datetime as dt
from keras.models import load_model
import streamlit as st

start= '2010-01-01'
end = '2021-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df= data.DataReader(user_input,'yahoo', start=start, end=end)

#describing data
st.subheader('Data from 2010-2021')
st.write(df.describe())

#visualisation
st.subheader('Closing Price vs Time chart with 100MA & 200 MA')
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, 'r', label='ma100')
plt.plot(ma200, 'g', label='ma200')
plt.plot(df.Close, 'b', label='Closing Price')
st.pyplot(fig)

#split data into train and test
data_train= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test= pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#scale data between 0-1
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_array= scaler.fit_transform(data_train)

#load .h5 model
model=load_model('keras_model.h5')


#now make predictions- feed data into model
past_100_days=data_train.tail(100)
final_df = past_100_days.append(data_test,ignore_index=True)
#need to apply scaling to test data
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[ i-100 : i ])
  y_test.append(input_data[ i, 0 ])
#convert to numpy array
x_test, y_test = np.array(x_test), np.array(y_test)
#predict
y_predicted = model.predict(x_test)
#scale up by scale factor
scaler =scaler.scale_
scale_factor= 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph

st.subheader('Predictions vs Original')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
