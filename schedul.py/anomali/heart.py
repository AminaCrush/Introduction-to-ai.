import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv("Heart_Disease_Prediction.csv")

# Удаление категориального столбца "Heart Disease" — это цель, а не признак
df = df.drop(columns=["Heart Disease"])

# Масштабирование числовых данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Метод локтя — определить оптимальное число кластеров
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Метод локтя")
plt.xlabel("Количество кластеров")
plt.ylabel("Инерция")
plt.grid()
plt.show()

# Обучение модели с выбранным числом кластеров (например, k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Добавление кластера в датафрейм
df['Cluster'] = clusters

# Визуализация по 2 признакам (например, Age и Max HR)
sns.scatterplot(data=df, x="Age", y="Max HR", hue="Cluster", palette="Set1")
plt.title("Кластеризация по Age и Max HR")
plt.show()

plt.title("Кластеризация по Age и Max HR")
plt.show()

print("Число строк:", len(df))
print("Распределение по кластерам:")
print(df['Cluster'].value_counts())
print("\nПервые 5 строк с метками кластеров:")
print(df.head())
