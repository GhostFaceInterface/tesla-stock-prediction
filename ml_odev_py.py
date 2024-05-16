#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st

# Veri setini yükleme
df = pd.read_csv("TSLA.csv")

# Streamlit başlığı
st.title("TSLA Stock Price Prediction")
st.write("This application uses machine learning models to predict TSLA stock closing prices.")

# Veriyi detaylı incelemek için oluşturulan bir fonksiyon
def check_df(df, head=5):
    st.write("Shape", df.shape)
    st.write("Types", df.dtypes)
    st.write("Head", df.head(head))
    st.write("Tail", df.tail(head))
    st.write("NA", df.isnull().sum())
    numeric_df = df.select_dtypes(include=['number'])
    st.write("Quantiles", numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Object olan Date değişkenini gerçek Date tipine dönüştürme işlemi 
df.Date = pd.to_datetime(df.Date)

tesla_df = df[["Date", "Close"]]

st.write("Başlangıç tarihi: ", tesla_df.Date.min())
st.write("En son tarih: ", tesla_df.Date.max())

# Tarih odaklı değişken tanımlaması için Date kolonunun index olarak dönüşüm geçirmesi gerekiyor
tesla_df.index = tesla_df.Date
tesla_df.drop("Date", axis=1, inplace=True)

# Grafik gösterimi
plt.figure(figsize=(12, 6))
plt.plot(tesla_df.Close, color="red")
plt.ylabel("Hisse fiyatı")
plt.title("Tarihe göre Tesla hisseleri")
plt.xlabel("Tarih")
st.pyplot(plt)

# İşlem yapabilmek için datasetini numpy array'e çevirmemiz gerek
tesla_df = tesla_df.values

correlation_matrix = df.corr()
st.write(correlation_matrix['Close'].sort_values(ascending=False))

# Varyans eşik değeri belirleme
selector = VarianceThreshold(threshold=0.01)
df_var = selector.fit_transform(df[['Open', 'High', 'Low', 'Volume']])
selected_features_var = df[['Open', 'High', 'Low', 'Volume']].columns[selector.get_support()]
st.write("Selected Features (Variance Threshold):", selected_features_var)

# Recursive Feature Elimination
model = LinearRegression()
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(df[['Open', 'High', 'Low', 'Volume']], df['Close'])
selected_features_rfe = df[['Open', 'High', 'Low', 'Volume']].columns[fit.support_]
st.write("Selected Features (RFE):", selected_features_rfe)

# Train-test ayrımı
def split_data(df, test_size):
    position = int(round(len(df) * (1 - test_size)))
    train = df[:position]
    test = df[position:]
    return train, test, position

train, test, position = split_data(tesla_df, 0.20)
st.write("Train shape:", train.shape, "Test shape:", test.shape)

# Veriyi normalize etme
scaler_train = MinMaxScaler(feature_range=(0, 1))
train = scaler_train.fit_transform(train)
scaler_test = MinMaxScaler(feature_range=(0, 1))
test = scaler_test.fit_transform(test)

# Veri setini periyotlar halinde bölme
def create_features(df, lookback):
    X, Y = [], []
    for i in range(lookback, len(df)):
        X.append(df[i-lookback:i, 0])
        Y.append(df[i, 0])
    return np.array(X), np.array(Y)

lookback = 20
X_train, y_train = create_features(train, lookback)
X_test, y_test = create_features(test, lookback)

st.write("X_train shape:", X_train.shape, "y_train shape:", y_train.shape, "X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

# Karar Ağacı Modeli
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
st.subheader("Decision Tree Model")
st.write("MSE: ", mean_squared_error(y_test, dt_predictions))
st.write("MAE: ", mean_absolute_error(y_test, dt_predictions))
st.write("R2: ", r2_score(y_test, dt_predictions))

# SVM Modeli
svm_model = SVR()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
st.subheader("SVM Model")
st.write("MSE: ", mean_squared_error(y_test, svm_predictions))
st.write("MAE: ", mean_absolute_error(y_test, svm_predictions))
st.write("R2: ", r2_score(y_test, svm_predictions))

# KNN Modeli
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
st.subheader("KNN Model")
st.write("MSE: ", mean_squared_error(y_test, knn_predictions))
st.write("MAE: ", mean_absolute_error(y_test, knn_predictions))
st.write("R2: ", r2_score(y_test, knn_predictions))

# Model tahminlerini grafik olarak gösterme
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label='True Prices')
plt.plot(range(len(dt_predictions)), dt_predictions, label='Decision Tree Predictions')
plt.plot(range(len(svm_predictions)), svm_predictions, label='SVM Predictions')
plt.plot(range(len(knn_predictions)), knn_predictions, label='KNN Predictions')
plt.xlabel('Time')
plt.ylabel('Normalized Price')
plt.legend()
plt.title('Model Predictions vs True Prices')
st.pyplot(plt)
