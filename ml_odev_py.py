import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# Streamlit başlığı
st.title("Makine Öğrenimi Model Performansı Karşılaştırma")

# Veri setini yükleyelim
uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Veri setini inceleyelim
    st.write("### Veri Seti Başlıkları")
    st.write(df.head())
    
    st.write("### Veri Seti Bilgileri")
    st.write(df.info())

    # Veriyi detaylı inceleme fonksiyonu
    def check_df(df, head=5):
        st.write("### Shape")
        st.write(df.shape)
        st.write("### Types")
        st.write(df.dtypes)
        st.write("### Head")
        st.write(df.head(head))
        st.write("### Tail")
        st.write(df.tail(head))
        st.write("### NA")
        st.write(df.isnull().sum())
        numeric_df = df.select_dtypes(include=['number'])  # Sadece sayısal sütunları seç
        st.write("### Quantiles")
        st.write(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
    check_df(df)

    # object olan Date değişkenini gerçek Date tipine dönüştürme işlemi
    df['Date'] = pd.to_datetime(df['Date'])
    tesla_df = df[["Date","Close"]]
    tesla_df.index =  tesla_df['Date']
    tesla_df.drop(columns=['Date'], inplace=True)

    st.write("### İşlenen Veri Seti")
    st.write(tesla_df.head())

    # Veri hazırlığı fonksiyonu
    def create_features(df, lookback):
        X, y = [], []
        for i in range(len(df) - lookback):
            X.append(df[i:(i + lookback), 0])
            y.append(df[i + lookback, 0])
        return np.array(X), np.array(y)

    # Veriyi ölçeklendirme
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(tesla_df.values)

    # Veriyi eğitim ve test setlerine ayırma
    train_size = int(len(scaled_data) * 0.8)
    train, test = scaled_data[:train_size], scaled_data[train_size:]

    lookback = 20
    X_train, y_train = create_features(train, lookback)
    X_test, y_test = create_features(test, lookback)

    st.write("### Eğitim ve Test Setlerinin Boyutları")
    st.write("X_train:", X_train.shape, "y_train:", y_train.shape)
    st.write("X_test:", X_test.shape, "y_test:", y_test.shape)

    # Karar Ağacı Modeli
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)

    st.write("### Decision Tree Modeli Sonuçları")
    st.write("MSE: ", mean_squared_error(y_test, dt_predictions))
    st.write("MAE: ", mean_absolute_error(y_test, dt_predictions))
    st.write("R2: ", r2_score(y_test, dt_predictions))

    # SVM Modeli
    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    st.write("### SVM Modeli Sonuçları")
    st.write("MSE: ", mean_squared_error(y_test, svm_predictions))
    st.write("MAE: ", mean_absolute_error(y_test, svm_predictions))
    st.write("R2: ", r2_score(y_test, svm_predictions))

    # KNN Modeli
    knn_model = KNeighborsRegressor()
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)

    st.write("### KNN Modeli Sonuçları")
    st.write("MSE: ", mean_squared_error(y_test, knn_predictions))
    st.write("MAE: ", mean_absolute_error(y_test, knn_predictions))
    st.write("R2: ", r2_score(y_test, knn_predictions))
