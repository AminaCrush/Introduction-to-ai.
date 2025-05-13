import pandas as pd   # РАБОТА ТАБЛИЦ
from sklearn.model_selection import train_test_split  #БИБЛИОТЕКА ДЛЯ МАШИННОГО ОБУЧЕНИЯ
from sklearn.feature_extraction.text import TfidfVectorizer # ТЕКСТ В ЧИСЛА 
from sklearn.ensemble import RandomForestClassifier # СПАМ НЕ СПАМ 
from sklearn.metrics import classification_report, accuracy_score #  report метрику (точность, полнота 
import joblib  #СОХРОНЕНИЕ МОДЕЛЕЙ


df = pd.read_csv("spam_assassin.csv")

X = df["text"]
y = df["target"]

#текст в числа 
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000) #  100 самых важных слов 
X_vec = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказания и оценка качества
y_pred = model.predict(X_test)
# Выводим точность модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Сохраняем модель и векторайзер
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
