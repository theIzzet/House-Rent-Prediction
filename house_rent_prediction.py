import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


rent_data=pd.read_csv('House_Rent_Dataset (1).csv')


rent_data = rent_data.drop(['Posted On','Area Locality','Floor'],axis=1)

# Ardından sadece sayısal sütunlar için korelasyon çiz
corr = rent_data.select_dtypes(include=[np.number]).corr()
plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Sayısal Değişkenler Arası Korelasyon Matrisi')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=rent_data['Rent'])
plt.title('Kira Değerleri İçin Boxplot (Aykırı Değerler)')
plt.xlabel('Kira')
plt.show()


print('UÇ DEĞERLER')
print(np.where(rent_data['Rent']>500000))

rent_data.drop([726,  792,  827, 1001, 1319, 1329, 1384, 1459, 1484, 1837, 2750,
       3656], axis=0, inplace=True)



rent_data = pd.get_dummies(rent_data, columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
rent_data.head()


X = rent_data.drop('Rent',axis=1)
y = rent_data['Rent']


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)



y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()


X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)



print(X_train,y_train)



lm = LinearRegression()
lm.fit(X_train,y_train)
lm_prediction = lm.predict(X_test)

# Evaluation metrics
mae_lm = metrics.mean_absolute_error(y_test, lm_prediction)
mse_lm =  metrics.mean_squared_error(y_test, lm_prediction)
rmse_lm =  np.sqrt(mse_lm)



print('MAE:', mae_lm)
print('MSE:', mse_lm)
print('RMSE:', rmse_lm)




dt = DecisionTreeRegressor(random_state = 100)
dt.fit(X_train, y_train)
dt_prediction = dt.predict(X_test)

# Evaluation metrics
mae_dt = metrics.mean_absolute_error(y_test, dt_prediction)
mse_dt =  metrics.mean_squared_error(y_test, dt_prediction)
rmse_dt =  np.sqrt(mse_dt)



print('MAE:', mae_dt)
print('MSE:', mse_dt)
print('RMSE:', rmse_dt)



from xgboost import XGBRegressor

# XGBoost modeli 
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train.ravel())  # ravel() ile y_train tek boyuta indirildi


xgb_prediction = xgb_model.predict(X_test)

# Hata metriklerinin hesaplanması
mae_xgb = mean_absolute_error(y_test, xgb_prediction)
mse_xgb = mean_squared_error(y_test, xgb_prediction)
rmse_xgb = np.sqrt(mse_xgb)


print('XGBoost Modeli Performansı:')
print('MAE:', mae_xgb)
print('MSE:', mse_xgb)
print('RMSE:', rmse_xgb)


# R2 Skorları
r2_lm = r2_score(y_test, lm_prediction)
r2_dt = r2_score(y_test, dt_prediction)
r2_xgb = r2_score(y_test, xgb_prediction)

print('\n📊 R² Değerleri:')
print(f'Linear Regression R²: {r2_lm:.4f}')
print(f'Decision Tree R²: {r2_dt:.4f}')
print(f'XGBoost R²: {r2_xgb:.4f}')






X_raw = rent_data.drop('Rent', axis=1)
y_raw = rent_data['Rent']

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)

# Decision Tree 
dt_raw = DecisionTreeRegressor(random_state=100)
dt_raw.fit(X_train_raw, y_train_raw)
dt_raw_pred = dt_raw.predict(X_test_raw)

mae_dt_raw = mean_absolute_error(y_test_raw, dt_raw_pred)
mse_dt_raw = mean_squared_error(y_test_raw, dt_raw_pred)
rmse_dt_raw = np.sqrt(mse_dt_raw)
r2_dt_raw = r2_score(y_test_raw, dt_raw_pred)


xgb_raw = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_raw.fit(X_train_raw, y_train_raw)
xgb_raw_pred = xgb_raw.predict(X_test_raw)

mae_xgb_raw = mean_absolute_error(y_test_raw, xgb_raw_pred)
mse_xgb_raw = mean_squared_error(y_test_raw, xgb_raw_pred)
rmse_xgb_raw = np.sqrt(mse_xgb_raw)
r2_xgb_raw = r2_score(y_test_raw, xgb_raw_pred)


print('\n📊 Ölçeklenmemiş Verilerle Performans (Gerçek Kira Üzerinden):')
print(f'Decision Tree (Raw) - MAE: {mae_dt_raw:.2f}, RMSE: {rmse_dt_raw:.2f}, R²: {r2_dt_raw:.4f}')
print(f'XGBoost (Raw)       - MAE: {mae_xgb_raw:.2f}, RMSE: {rmse_xgb_raw:.2f}, R²: {r2_xgb_raw:.4f}')




from sklearn.ensemble import RandomForestRegressor


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.ravel())
rf_prediction = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, rf_prediction)
mse_rf = mean_squared_error(y_test, rf_prediction)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, rf_prediction)

print('\nRandom Forest Modeli Performansı:')
print(f'MAE: {mae_rf:.4f}')
print(f'MSE: {mse_rf:.4f}')
print(f'RMSE: {rmse_rf:.4f}')
print(f'R²: {r2_rf:.4f}')



rf_raw = RandomForestRegressor(n_estimators=100, random_state=42)
rf_raw.fit(X_train_raw, y_train_raw)
rf_raw_pred = rf_raw.predict(X_test_raw)

mae_rf_raw = mean_absolute_error(y_test_raw, rf_raw_pred)
mse_rf_raw = mean_squared_error(y_test_raw, rf_raw_pred)
rmse_rf_raw = np.sqrt(mse_rf_raw)
r2_rf_raw = r2_score(y_test_raw, rf_raw_pred)

print('\nRandom Forest (Raw) Performansı:')
print(f'MAE: {mae_rf_raw:.2f}, RMSE: {rmse_rf_raw:.2f}, R²: {r2_rf_raw:.4f}')



results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'MAE': [mae_lm, mae_dt, mae_rf, mae_xgb],
    'RMSE': [rmse_lm, rmse_dt, rmse_rf, rmse_xgb],
    'R²': [r2_lm, r2_dt, r2_rf, r2_xgb]
})

results_df_raw = pd.DataFrame({
    'Model (Raw Data)': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'MAE': [mae_dt_raw, mae_rf_raw, mae_xgb_raw],
    'RMSE': [rmse_dt_raw, rmse_rf_raw, rmse_xgb_raw],
    'R²': [r2_dt_raw, r2_rf_raw, r2_xgb_raw]
})

print('\n📊 Ölçeklenmiş Verilerle Model Performans Karşılaştırması:')
print(results_df)

print('\n📊 Ölçeklenmemiş Verilerle Model Performans Karşılaştırması:')
print(results_df_raw)
models = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest', 'XGBoost']

# Metirkler
mae_values = [mae_lm, mae_dt, mae_rf, mae_xgb]
rmse_values = [rmse_lm, rmse_dt, rmse_rf, rmse_xgb]
r2_values = [r2_lm, r2_dt, r2_rf, r2_xgb]

x = np.arange(len(models))  # Model konumları
width = 0.25  # Bar genişliği

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(x - width, mae_values, width, label='MAE')
plt.bar(x, rmse_values, width, label='RMSE')
plt.bar(x + width, r2_values, width, label='R²')

plt.xticks(x, models)
plt.ylabel('Skor')
plt.title('Model Performans Karşılaştırması (Scaled Data)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


