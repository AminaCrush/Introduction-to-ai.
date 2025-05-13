import pandas as pd
import matplotlib.pyplot as plt  # для построение графиков 
from sklearn.cluster import KMeans # алгоритм \\ kmeans разделяет на группы
from sklearn.preprocessing import StandardScaler # среднее = 0, стандартное отклонение = 1

# Загрузка данных
data = pd.read_csv("Mall_Customers.csv")
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
num_clusters = 5
# обучаем
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled) # находит центр класстеров 

# кто в каком кластере.
data["Cluster"] = clusters

# Визуализация кластеров
plt.figure(figsize=(8,6))
for i in range(num_clusters):
    plt.scatter(                            #рисует точки 
        X_scaled[clusters == i, 0],  #   ось X (годовой доход, стандартизованный
        X_scaled[clusters == i, 1],  #  ось Y (оценка трат)
        label=f"Кластер {i}"
    )
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, label='Центры')
plt.title("Сегментация клиентов")
plt.xlabel("Годовой доход (стандартизованный)")
plt.ylabel("Оценка расходов (стандартизованная)")
plt.legend()
plt.show()



#plt.legend() — показывает, какой цвет какому кластеру соответствует.
#plt.show() — выводит график на экран.




#Кластер 0 (оранжевый)
#→ 💰 Высокий доход, но 💳 низкие траты
#Это могут быть бережливые или рациональные клиенты.

#Кластер 1 (зелёный)
#→ 💰 Низкий доход, но 💳 высокие траты
#Это импульсивные покупатели или молодые клиенты.

#Кластер 2 (голубой)
#→ 💰 Средний доход и 💳 средние траты
#Это стабильные, сбалансированные клиенты.