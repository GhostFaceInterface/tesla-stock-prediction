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
import matplotlib.pyplot as plt

# Fonksiyonlar
def check_df(df, head=5):
    st.write("### Veri Seti Bilgileri")
    st.write("#### Şekil")
    st.write(df.shape)
    st.write("#### Türler")
    st.write(df.dtypes)
    st.write("#### İlk Satırlar")
    st.write(df.head(head))
    st.write("#### Son Satırlar")
    st.write(df.tail(head))
    st.write("#### Eksik Değerler")
    st.write(df.isnull().sum())
    st.write("#### Quantiles")
    numeric_df = df.select_dtypes(include=['number'])  # Sadece sayısal sütunları seç
    st.write(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)    

def split_data(df, test_size):
    position = int(round(len(df) * (1 - test_size)))
    train = df[:position]
    test = df[position:]
    return train, test

def create_features(df, lookback):
    X, Y = [], []
    for i in range(lookback, len(df)):
        X.append(df[i-lookback:i, :-1].flatten())  # Özellikler (Open, High, Low)
        Y.append(df[i, -1])  # Hedef değişken (Close)
    return np.array(X), np.array(Y)

# Streamlit başlatma
st.title("Hisse Senedi Fiyat Tahmini")
st.write("Bu uygulama, Tesla hisse senedi fiyatını tahmin etmek için makine öğrenmesi modellerini kullanır.")

# Veri yükleme
uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    check_df(df)

    # Tarih tipine dönüştürme
    df['Date'] = pd.to_datetime(df['Date'])

    # Grafik gösterimi
    st.write("### Tarihe Göre Tesla Hisse Fiyatları")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], color="red")
    plt.ylabel("Hisse Fiyatı")
    plt.title("Tarihe Göre Tesla Hisseleri")
    plt.xlabel("Tarih")
    st.pyplot(plt)

    # Özellik seçimi
    selected_features = ['Open', 'High', 'Low', 'Close']
    tesla_df = df[['Date'] + selected_features]
    tesla_df.set_index('Date', inplace=True)
    
    # Train-test ayrımı
    train, test = split_data(tesla_df, 0.20)

    # Normalizasyon işlemi
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # Veri setini periyotlar halinde bölme
    lookback = st.slider("Lookback Period", min_value=1, max_value=60, value=20)
    X_train, y_train = create_features(train_scaled, lookback)
    X_test, y_test = create_features(test_scaled, lookback)

    # Model seçimi
    model_option = st.selectbox("Model Seçin", ["Karar Ağacı", "SVM", "KNN"])

    if model_option == "Karar Ağacı":
        model = DecisionTreeRegressor()
    elif model_option == "SVM":
        model = SVR()
    elif model_option == "KNN":
        model = KNeighborsRegressor()

    # Model eğitimi
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Sonuçların gösterimi
    st.write(f"### {model_option} Model Sonuçları")
    st.write("Mean Squared Error (MSE): ", mean_squared_error(y_test, predictions))
    st.write("Mean Absolute Error (MAE): ", mean_absolute_error(y_test, predictions))
    st.write("R-squared (R²): ", r2_score(y_test, predictions))
