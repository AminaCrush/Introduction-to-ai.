import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Загрузка данных
train_df = pd.read_csv('Hous_onePrices.csv')

# Доступные признаки
features = ['Area', 'Room', 'Lon', 'Lat']

# Удаление пропусков
train_df = train_df.dropna(subset=['Price'] + features)

# Инженерия признаков (добавляем взаимодействие)
train_df['Area_Room'] = train_df['Area'] * train_df['Room']
features.append('Area_Room')  # Добавляем новый признак

X = train_df[features]
y = np.log1p(train_df['Price'])  # Логарифмирование для стабильности

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ОБУЧАЕМ МОДЕЛЬ 
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Random Forest (более мощная модель)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Random Forest не требует масштабирования
y_pred_rf = rf_model.predict(X_test)

# Метрики для линейной регрессии
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred_lr)) / np.expm1(y_test))) * 100

# Метрики для Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred_rf)) / np.expm1(y_test))) * 100

# Вывод результатов
print("\nМетрики для Линейной регрессии:")
print(f"MSE: {mse_lr:.4f}")
print(f"MAE: {mae_lr:.4f}")
print(f"R²: {r2_lr:.4f}")
print(f"MAPE: {mape_lr:.2f}%")

print("\nМетрики для Random Forest:")
print(f"MSE: {mse_rf:.4f}")
print(f"MAE: {mae_rf:.4f}")
print(f"R²: {r2_rf:.4f}")
print(f"MAPE: {mape_rf:.2f}%")

# График для Random Forest (используем его, если R² выше)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Реальные значения (log)")
plt.ylabel("Предсказанные значения (log)")
plt.title(f"Random Forest (R² = {r2_rf:.2f})")
plt.grid(True)

# Сохранение графика
plt.savefig('scatter_plot_rf_amsterdam.png')
plt.show()

# Проверка условия R² ≥ 0.9
if r2_rf >= 0.9:
    print("Точность R² ≥ 0.9 достигнута с Random Forest!")
else:
    print("Точность R² < 0.9. Нужно больше признаков или данных для достижения 90%.")