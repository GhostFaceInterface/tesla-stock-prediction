#!/usr/bin/env python
# coding: utf-8

# In[57]:


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




# In[58]:


df = pd.read_csv("TSLA.csv")



# In[59]:


df.head()


# In[ ]:





# In[60]:


#veriyi detaylı incelemek için oluşturulan bir fonksiyon
def check_df(df, head = 5):
    print("#################### Shape ####################")
    print(df.shape)
    print("#################### Types ####################")
    print(df.dtypes)
    print("#################### Head ####################")
    print(df.head(head))
    print("#################### Tail ####################")
    print(df.tail(head))
    print("#################### NA ####################")
    print(df.isnull().sum())
    print("#################### Quantiles ####################")
    numeric_df = df.select_dtypes(include=['number'])  # Sadece sayısal sütunları seç
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)    
    


# In[61]:


check_df(df)


# In[62]:


#object olan Date değişkenini gerçek Date tipine dönüştürme işlemi 
df.Date = pd.to_datetime(df.Date)


# In[63]:


df.head()


# In[64]:


tesla_df = df[["Date","Close"]]


# In[65]:


tesla_df.head()


# In[66]:


print(" Başlangıç tarihi : ", tesla_df.Date.min())
print(" En son tarih : ", tesla_df.Date.max())


# In[67]:


#Tarih odaklı değişken tanımlaması için Date kolonunun index olarak dönüşüm geçirmesi gerekiyor
tesla_df.index =  tesla_df.Date


# In[68]:


tesla_df


# In[69]:


tesla_df.drop("Date", axis=1,inplace = True)


# In[70]:


tesla_df


# In[71]:


copy_df = tesla_df.copy()


# In[72]:


plt.figure(figsize=(12,6))
plt.plot(tesla_df.Close, color = "red")
plt.ylabel("Hisse fiyatı")
plt.title("Tarihe göre Tesla hisseleri")
plt.xlabel("Tarih")
plt.show()


# In[73]:


#işlem yapabilmek için datasetini numpy array'e çevirmemiz gerek
tesla_df = tesla_df.values


# In[74]:


tesla_df[0:10]


# In[ ]:





# In[75]:


correlation_matrix = df.corr()
print(correlation_matrix['Close'].sort_values(ascending=False))


# In[76]:


from sklearn.feature_selection import VarianceThreshold

# Varyans eşik değeri belirleme
selector = VarianceThreshold(threshold=0.01)
df_var = selector.fit_transform(df[['Open', 'High', 'Low', 'Volume']])
selected_features_var = df[['Open', 'High', 'Low', 'Volume']].columns[selector.get_support()]
print(selected_features_var)


# In[77]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
#Recursive Feature Elimination
model = LinearRegression()
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(df[['Open', 'High', 'Low', 'Volume']], df['Close'])
selected_features_rfe = df[['Open', 'High', 'Low', 'Volume']].columns[fit.support_]
print(selected_features_rfe)


# In[78]:


#buradan da anlaşılacağı üzere (['Open', 'High', 'Low']) bizim için ideal özellik seçimi olacaktır


# In[79]:


#train-test ayrımını scikit-learn ile ayırmamız gerekirdi ama zamana bağlı indeksleme yaptığımız için train-test ayrımını kendimiz yapmamız gerekiyor çünkü scikit-learn rastgele seçimler yapıyor
def spit_data(df, test_size):
    position = int(round(len(df) * (1 - test_size)))
    train =  df[:position]
    test = df[position:]
    return train, test, position 


# In[80]:


train, test, position = spit_data(tesla_df, 0.20)


# In[81]:


print(train.shape, test.shape)


# In[82]:


scaler_train = MinMaxScaler(feature_range=(0,1))
train = scaler_train.fit_transform(train)


# In[83]:


scaler_test = MinMaxScaler(feature_range=(0,1))
test = scaler_test.fit_transform(test)


# In[84]:


train[0:5]


# In[85]:


test[0:5]


# In[86]:


#veri setini periyotlar halinde bölmemiz gerekiyor çünkü zamanla ilgili tahmin işlemi yapıldığında verinin önceki
#dönemlerinin de kullanılması gerekiyor
def create_features (df, lookback):
    X,Y =[], []
    for i in range (lookback, len(df)):
        X.append(df[i-lookback:i,0])
        Y.append(df[i,0])
    return np.array(X), np.array(Y)
                      


# In[87]:


lookback = 20


# In[88]:


X_train, y_train = create_features(train, lookback)
X_test, y_test = create_features(test, lookback)


# In[89]:


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[90]:


X_train[0:5]


# In[91]:


y_train[0:5]


# In[92]:


# Karar Ağacı Modeli
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

print("Decision Tree MSE: ", mean_squared_error(y_test, dt_predictions))
print("Decision Tree MAE: ", mean_absolute_error(y_test, dt_predictions))
print("Decision Tree R2: ", r2_score(y_test, dt_predictions))


# In[93]:


# SVM Modeli
svm_model = SVR()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

print("SVM MSE: ", mean_squared_error(y_test, svm_predictions))
print("SVM MAE: ", mean_absolute_error(y_test, svm_predictions))
print("SVM R2: ", r2_score(y_test, svm_predictions))


# In[94]:


# KNN Modeli
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

print("KNN MSE: ", mean_squared_error(y_test, knn_predictions))
print("KNN MAE: ", mean_absolute_error(y_test, knn_predictions))
print("KNN R2: ", r2_score(y_test, knn_predictions))


# In[95]:


#karışıklık matrisi (confusion matrix) genellikle sınıflandırma problemleri için kullanılır ve regresyon problemleri için uygun bir değerlendirme metriği değildir.
#SVM modeli en iyi performansı gösteriyor. MSE ve MAE değerleri en düşük, R2 skoru ise en yüksek olan modeldir.


# In[ ]:




