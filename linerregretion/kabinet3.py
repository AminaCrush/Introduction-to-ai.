import numpy as np  # для работы с числами  (ГЕНЕРАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ)
import pandas as pd  #для работы с таблицами
import matplotlib.pyplot as plt  # ПОСТРОЕНИЕ ГРАФИКА (figure)
import seaborn as sns  # (0.6)
from sklearn.linear_model import LinearRegression  # обучаем модель
from sklearn.model_selection import train_test_split  # тренируем и тестируем
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # точность предсказания
from sklearn.preprocessing import StandardScaler  # ОБЕДЕНЯЕТ И НАХОДИТ СРЕДНЕЕ ЗНАЧЕНИЕ

# НАМПАЙ
n_samples = 100  # Увеличим количество данных для стабильности

# Признаки одежды
size = np.random.uniform(36, 50, n_samples)  # Размер одежды (от 36 до 50)
details = np.random.randint(1, 6, n_samples)  # Кол-во декоративных элементов (1-5)
brand_rating = np.random.uniform(1.0, 5.0, n_samples)  # Рейтинг бренда (1-5)
season_code = np.random.randint(0, 4, n_samples)  # Код сезона (0-зима, 1-весна и т.п.)

# Цена как линейная комбинация признаков + небольшой шум
price = (8 * size) + (30 * details) + (100 * brand_rating) + (20 * season_code) + np.random.normal(0, 10, n_samples)

# Пандас
data = {
    'Price': price,

    'Size': size,
    'Details': details,
    'Brand': brand_rating,
    'Season': season_code
}
train_df = pd.DataFrame(data)